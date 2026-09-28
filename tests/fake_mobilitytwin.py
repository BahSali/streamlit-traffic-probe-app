"""Fake MobilityTwin 'parquetized' API, shaped like the real one in the deployed logs.

Older UTC days come back as one whole-day file for any window on that day;
today comes back as hourly files. Every link carries a fresh (fake) signature,
like a pre-signed download URL. File content is synthetic and deterministic.
"""
import io, re, uuid
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

TODAY_UTC = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
CALLS = {"list": 0, "download": 0}
_BYTES = {}
PAIRS = None

def set_pairs(lookup, n=150):
    global PAIRS
    PAIRS = lookup.drop_duplicates(["pointId", "lineId"]).head(n)[["lineId", "pointId"]].to_records(index=False)

def _file_table(start, end, seed):
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="20s", inclusive="left")
    frames = []
    for k, (line, point) in enumerate(PAIRS):
        step = 60 + 60 * rng.random(len(times))             # 3-18 m/s over 20 s
        frames.append(pd.DataFrame({"lineId": str(line), "pointId": int(point), "directionId": 1 + k % 2,
                                    "distanceFromPoint": np.cumsum(step) % 5000.0,
                                    "date": times + pd.to_timedelta(rng.integers(0, 3, len(times)), unit="s")}))
    return pa.Table.from_pandas(pd.concat(frames, ignore_index=True), preserve_index=False)

def _file_bytes(name):
    if name not in _BYTES:
        kind, stamp = name.split("_", 1)
        start = pd.Timestamp(stamp)
        end = start + (pd.Timedelta(days=1) if kind == "day" else pd.Timedelta(hours=1))
        buf = io.BytesIO(); pq.write_table(_file_table(start, end, int.from_bytes(name.encode()[-8:], 'little') % 2**32), buf)
        _BYTES[name] = buf.getvalue()
    return _BYTES[name]

def files_for(start_utc, end_utc):
    names = []
    day = start_utc.normalize()
    while day <= end_utc:
        if day < TODAY_UTC:
            names.append(f"day_{day:%Y-%m-%d}")
        else:
            hour = max(start_utc.floor("h"), day)
            while hour <= min(end_utc, pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(hours=1)):
                names.append(f"hour_{hour:%Y-%m-%dT%H}")
                hour += pd.Timedelta(hours=1)
        day += pd.Timedelta(days=1)
    return names

class _Resp:
    def __init__(self, payload=None, content=b""): self._p, self.content = payload, content
    def raise_for_status(self): pass
    def json(self): return self._p

def fake_get(url, headers=None, timeout=None, **kw):
    if "parquetized" in url:
        CALLS["list"] += 1
        s = float(re.search(r"start_timestamp=([\d.]+)", url).group(1)); e = float(re.search(r"end_timestamp=([\d.]+)", url).group(1))
        names = files_for(pd.Timestamp(s, unit="s"), pd.Timestamp(e, unit="s"))
        return _Resp({"results": [f"https://files.fake/stib/{n}.parquet?X-Amz-Expires=900&X-Amz-Signature={uuid.uuid4().hex}" for n in names]})
    CALLS["download"] += 1
    name = re.search(r"/stib/([^?]+)\.parquet", url).group(1)
    return _Resp(content=_file_bytes(name))
