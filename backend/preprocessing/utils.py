"""
Small, dependency-light utilities used across Part 3:

  - compute_content_hash: streamed sha256, used for content-hash-keyed
    image_id / storage paths (cross-cutting requirement: never filename-keyed).
  - sanitize_string: first-pass cleanup of any user-supplied string (declared
    timestamp, filename, metadata tag) before it can reach an LLM prompt
    downstream (cross-cutting requirement).
  - run_isolated: runs a callable in a separate process with a hard wall-clock
    timeout, so a corrupt or adversarial file can only ever fail its own job,
    never hang or crash the caller (Part 3 hardening requirement).

Deliberately free of rasterio/pydantic imports so this module — and its
behavior — can be exercised with nothing but the standard library plus numpy.
"""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import re
from dataclasses import dataclass
from typing import Any, Callable

from . import config


# --------------------------------------------------------------------------
# Content hashing
# --------------------------------------------------------------------------

def compute_content_hash(file_path: str, chunk_size: int = 1024 * 1024) -> str:
    """
    Streamed sha256 over a file's bytes. Reads in fixed-size chunks so hashing
    a multi-GB raster never requires holding the whole file in memory —
    consistent with the "windowed reads only" spirit applied to file I/O
    generally, not just pixel arrays.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_key_hash(*parts: Any, length: int = 16) -> str:
    """
    Deterministic short hash of arbitrary parameters (e.g. image_id + task +
    tile offsets), used for keys like tile_id. Not a content hash of pixel
    data — a stable hash of the parameters that define an artifact, so the
    same request always resolves to the same key without being filename-based.
    """
    joined = "|".join(str(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:length]


# --------------------------------------------------------------------------
# String sanitization
# --------------------------------------------------------------------------

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_WHITESPACE_RUN_RE = re.compile(r"\s+")
# Note: \t \n \r are deliberately excluded from _CONTROL_CHARS_RE — they're
# real whitespace, not control noise, and stripping them here (before the
# injection-pattern pass below) would erase the newline that the line-start
# role-spoofing patterns rely on to detect an attack. They're normalized to
# single spaces by _WHITESPACE_RUN_RE afterward instead.

# Patterns that could be used to break out of a templated LLM prompt and
# inject fake instructions or spoofed role turns further downstream (Part 5
# builds prose from evidence via an LLM call). This is a first line of
# defense applied at the point user text first enters the system — it does
# not replace treating all user-supplied text as inert data at the prompt
# assembly layer itself.
_INJECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"```"), "'''"),
    (re.compile(r"<\|"), "< |"),
    (re.compile(r"\|>"), "| >"),
    (re.compile(r"^\s*(system|assistant|user)\s*:", re.IGNORECASE | re.MULTILINE), "[field]:"),
]


def sanitize_string(value: str | None, max_len: int | None = None) -> str | None:
    """
    Cleans a single user-supplied string:
      1. Strips C0/C1 control characters.
      2. Collapses whitespace runs to a single space and trims ends.
      3. Neutralizes a small set of prompt-injection-shaped patterns.
      4. Truncates to max_len.

    None passes through as None (declared_timestamp is optional).
    """
    if value is None:
        return None
    if max_len is None:
        max_len = config.MAX_SANITIZED_STRING_LEN

    cleaned = _CONTROL_CHARS_RE.sub("", value)
    # Injection patterns are checked BEFORE whitespace collapse: the classic
    # attack shape is an embedded newline used to start a fake new "line" that
    # a prompt template would render as a fresh role turn. Collapsing
    # whitespace first would erase that newline and blind the line-start
    # patterns below to exactly the case they exist to catch.
    for pattern, replacement in _INJECTION_PATTERNS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = _WHITESPACE_RUN_RE.sub(" ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


# --------------------------------------------------------------------------
# Isolated, timeout-guarded execution
# --------------------------------------------------------------------------

@dataclass
class IsolatedResult:
    ok: bool
    value: Any = None
    error: str | None = None
    timed_out: bool = False


def _isolated_worker(queue: "mp.Queue", func: Callable, args: tuple, kwargs: dict) -> None:
    try:
        result = func(*args, **kwargs)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure must be captured, not raised
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_isolated(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    timeout: float | None = None,
) -> IsolatedResult:
    """
    Runs func(*args, **kwargs) in a separate process with a hard timeout.

    `func` must be a module-level function (picklable) — this uses the
    "spawn" start method deliberately, rather than "fork", so the worker
    doesn't inherit any open file handles / GDAL state from the parent.

    On timeout, the worker process is terminated (SIGTERM then SIGKILL if it
    doesn't die) and IsolatedResult(ok=False, timed_out=True) is returned.
    A corrupt file that hangs GDAL's parser therefore only ever costs one
    job's timeout budget, never the caller.
    """
    if kwargs is None:
        kwargs = {}
    if timeout is None:
        timeout = config.PARSE_TIMEOUT_SECONDS

    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_isolated_worker, args=(result_queue, func, args, kwargs))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return IsolatedResult(ok=False, error=f"parsing timed out after {timeout}s", timed_out=True)

    if not result_queue.empty():
        status, payload = result_queue.get()
        if status == "ok":
            return IsolatedResult(ok=True, value=payload)
        return IsolatedResult(ok=False, error=str(payload))

    # Process exited without putting anything on the queue at all — e.g. a
    # native crash / segfault inside a C extension, not a catchable Python
    # exception. Still must not propagate as a hang or an unhandled crash.
    return IsolatedResult(
        ok=False,
        error=f"worker process exited unexpectedly (exit code {proc.exitcode})",
    )
