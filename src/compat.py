"""Runtime compatibility helpers.

This machine's Windows Application Control policy blocks the compiled
``_xxhash`` extension. ``langchain-core`` imports ``langsmith`` at import time
and ``langsmith`` imports ``xxhash`` eagerly, even though it only uses it for
deterministic UUID derivation in tracing.

``ensure_xxhash()`` installs a tiny deterministic stand-in **only** when the
real module cannot be imported, so the rest of the pipeline keeps working.
Nothing in this project uses LangSmith tracing, so the substitution never
affects retrieval or answers.
"""

from __future__ import annotations

import hashlib
import sys
import types
import warnings


class _Xxh3_128:
    """Minimal xxh3_128 replacement: stable 16-byte BLAKE2b digest."""

    def __init__(self, data: bytes = b"") -> None:
        self._digest = hashlib.blake2b(data, digest_size=16).digest()

    def digest(self) -> bytes:
        return self._digest

    def hexdigest(self) -> str:
        return self._digest.hex()

    def intdigest(self) -> int:
        return int.from_bytes(self._digest, "big")


def ensure_xxhash() -> None:
    """Provide an ``xxhash`` module if the compiled extension is unavailable."""
    if "xxhash" in sys.modules:
        return
    try:
        import xxhash  # noqa: F401
    except ImportError:
        module = types.ModuleType("xxhash")
        module.__doc__ = "Fallback installed by aws-rag (compiled xxhash is blocked)."
        module.xxh3_128 = _Xxh3_128  # type: ignore[attr-defined]
        module.xxh128 = _Xxh3_128  # type: ignore[attr-defined]
        module.VERSION = "0.0.0-fallback"  # type: ignore[attr-defined]
        sys.modules["xxhash"] = module
        warnings.warn(
            "The compiled xxhash extension is blocked by Application Control; "
            "using a deterministic BLAKE2b fallback for LangSmith UUIDs.",
            RuntimeWarning,
            stacklevel=2,
        )
