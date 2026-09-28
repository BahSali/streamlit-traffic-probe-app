"""One RUN's update, computed in a background thread owned by the browser session.

Why a thread: any rerun of the page (a filter edit, a second click, a
reconnect) stops the script run that started the work. If the work ran in
the script itself, the next run would start it again, so one session could
have several copies of the same update running at once. Here the work keeps
running across reruns and every script run just shows its progress.

Rules that keep this safe:
- The thread gets all its inputs when it starts and never reads or writes
  st.session_state (it may outlive the script run that started it).
- Its result is applied to the session exactly once (apply_once), even if
  two script runs of the same session see it finish at the same moment.
- There is one job per browser session; other sessions are not blocked.
  Identical data needed by two sessions at once is computed once by the
  shared st.cache_data caches, which lock per cache key.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("estimator.brussels")


class UpdateJob:
    def __init__(self, work: Callable[[dict, Callable[[str], None]], Any], inputs: dict):
        self.inputs = inputs
        self.stages: list[str] = []  # stage labels reported so far, in order
        self.result: Any = None
        self.error: BaseException | None = None
        self.started_at = time.time()
        self._work = work
        self._done = threading.Event()
        self._apply_lock = threading.Lock()
        self._applied = False
        self._thread = threading.Thread(target=self._run, name="brussels-update", daemon=True)

    def start(self) -> "UpdateJob":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.result = self._work(self.inputs, self.stages.append)
        except BaseException as exc:  # shown on the page by the script run that applies it
            logger.exception("Brussels update failed")
            self.error = exc
        finally:
            self._done.set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def apply_once(self, apply: Callable[["UpdateJob"], None]) -> None:
        """Run apply(self) if no other script run has; returns after it has run."""
        with self._apply_lock:
            if not self._applied:
                apply(self)
                self._applied = True
