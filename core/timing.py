"""Stage timings for developers, written to the server log (never the page).

    with timed("brussels.google_routes", requests=12):
        ...

logs a line such as ``timing brussels.google_routes 1.234s requests=12``.
Set ESTIMATOR_TIMING_LOG=0 to silence it.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager

logger = logging.getLogger("estimator.timing")

if not logger.handlers:
    _handler = logging.StreamHandler(sys.stderr)
    _handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO if os.environ.get("ESTIMATOR_TIMING_LOG", "1") != "0" else logging.WARNING)


@contextmanager
def timed(stage: str, **fields):
    """Log how long the block took. Extra fields can be added to the yielded dict."""
    extra: dict = dict(fields)
    start = time.perf_counter()
    try:
        yield extra
    finally:
        elapsed = time.perf_counter() - start
        details = " ".join(f"{key}={value}" for key, value in extra.items())
        logger.info("timing %s %.3fs %s", stage, elapsed, details)
