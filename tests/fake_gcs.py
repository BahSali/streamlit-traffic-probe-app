"""Stand-ins for a Google Cloud Storage bucket with generation preconditions.

They implement only what GcsUsageStore uses, with GCS's documented rule: a
read or write with ``if_generation_match`` fails with PreconditionFailed
(HTTP 412) unless the object's current generation matches (0 = must not
exist). The random delays widen race windows between reading and writing.
"""
from __future__ import annotations

import fcntl
import json
import random
import threading
import time
from pathlib import Path

from google.api_core.exceptions import PreconditionFailed


def _jitter(max_delay: float) -> None:
    if max_delay:
        time.sleep(random.uniform(0, max_delay))


class _Blob:
    def __init__(self, bucket, name: str, generation: int | None = None) -> None:
        self.bucket, self.name, self.generation = bucket, name, generation

    def download_as_bytes(self, if_generation_match=None) -> bytes:
        _jitter(self.bucket.max_delay)
        return self.bucket._read(self.name, if_generation_match)

    def upload_from_string(self, data, content_type=None, if_generation_match=None) -> None:
        _jitter(self.bucket.max_delay)
        self.bucket._write(self.name, data if isinstance(data, bytes) else data.encode(), if_generation_match)


class MemoryBucket:
    """Thread-safe in-memory bucket (one process, many threads)."""

    def __init__(self, max_delay: float = 0.0) -> None:
        self.objects: dict[str, tuple[bytes, int]] = {}
        self.lock = threading.Lock()
        self.max_delay = max_delay
        self.writes = 0

    def get_blob(self, name):
        with self.lock:
            if name not in self.objects:
                return None
            return _Blob(self, name, self.objects[name][1])

    def blob(self, name):
        return _Blob(self, name)

    def _read(self, name, if_generation_match):
        with self.lock:
            data, generation = self.objects[name]
            if if_generation_match is not None and generation != if_generation_match:
                raise PreconditionFailed("generation mismatch")
            return data

    def _write(self, name, data, if_generation_match):
        with self.lock:
            current = self.objects.get(name, (b"", 0))[1]
            if if_generation_match is not None and current != if_generation_match:
                raise PreconditionFailed("generation mismatch")
            self.objects[name] = (data, current + 1)
            self.writes += 1

    def document(self, name) -> dict:
        return json.loads(self.objects[name][0])


class FileBucket:
    """Bucket in a directory, shared by several processes (an OS file lock
    makes each compare-and-set step atomic, as GCS does server-side)."""

    def __init__(self, root: Path, max_delay: float = 0.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_delay = max_delay

    def _locked(self):
        handle = open(self.root / ".lock", "a+")
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle

    def _paths(self, name):
        safe = name.replace("/", "__")
        return self.root / safe, self.root / (safe + ".generation")

    def _generation(self, name) -> int:
        _, gen_path = self._paths(name)
        return int(gen_path.read_text()) if gen_path.exists() else 0

    def get_blob(self, name):
        with self._locked():
            generation = self._generation(name)
        return None if generation == 0 else _Blob(self, name, generation)

    def blob(self, name):
        return _Blob(self, name)

    def _read(self, name, if_generation_match):
        with self._locked():
            if if_generation_match is not None and self._generation(name) != if_generation_match:
                raise PreconditionFailed("generation mismatch")
            return self._paths(name)[0].read_bytes()

    def _write(self, name, data, if_generation_match):
        with self._locked():
            current = self._generation(name)
            if if_generation_match is not None and current != if_generation_match:
                raise PreconditionFailed("generation mismatch")
            data_path, gen_path = self._paths(name)
            data_path.write_bytes(data)
            gen_path.write_text(str(current + 1))
