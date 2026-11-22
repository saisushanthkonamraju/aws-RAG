"""AWS whitepaper RAG package.

Importing this package installs the xxhash compatibility shim (see
``src.compat``) before any LangChain module can import it.
"""

from src.compat import ensure_xxhash

ensure_xxhash()
