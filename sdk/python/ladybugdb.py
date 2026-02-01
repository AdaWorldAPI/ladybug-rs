"""
LadybugDB Python SDK
====================

Unified client for LadybugDB cognitive database.
Works with both:
  1. HTTP REST API (ladybug-server running on Railway/Docker)
  2. Native PyO3 bindings (import ladybug — compiled .so)

LanceDB-Compatible API:
  db = ladybugdb.connect("http://localhost:8080")
  table = db.create_table("thoughts", data=[...])
  results = table.search("query text").limit(10).to_list()

Low-Level API:
  client = ladybugdb.Client("http://localhost:8080")
  fp = client.fingerprint("hello world")
  results = client.topk(fp, k=10)
  dist = client.hamming(fp1, fp2)

Installation:
  pip install requests  # only dependency for HTTP mode
  # OR compile native: maturin develop --features python
"""

from __future__ import annotations
import json
import base64
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import ladybug as _native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


__version__ = "0.2.0"
FINGERPRINT_BITS = 10_000
FINGERPRINT_BYTES = 1_256  # 157 × 8


# =============================================================================
# HTTP CLIENT (REST API)
# =============================================================================

class Client:
    """Low-level HTTP client for LadybugDB REST API."""

    def __init__(self, url: str = "http://localhost:8080", timeout: float = 30.0):
        if not HAS_REQUESTS:
            raise ImportError("requests library required: pip install requests")
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    def _post(self, path: str, data: dict) -> dict:
        r = self._session.post(f"{self.url}{path}", json=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = self._session.get(f"{self.url}{path}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # --- Info ---
    def health(self) -> dict:
        return self._get("/health")

    def info(self) -> dict:
        return self._get("/api/v1/info")

    def simd(self) -> dict:
        return self._get("/api/v1/simd")

    # --- Fingerprint operations ---
    def fingerprint(self, content: str) -> str:
        """Create fingerprint from content. Returns base64-encoded fingerprint."""
        r = self._post("/api/v1/fingerprint", {"content": content})
        return r["fingerprint"]

    def fingerprint_random(self) -> str:
        """Create random fingerprint."""
        r = self._post("/api/v1/fingerprint", {})
        return r["fingerprint"]

    def fingerprint_batch(self, contents: List[str]) -> List[str]:
        """Create fingerprints for multiple contents."""
        r = self._post("/api/v1/fingerprint/batch", {"contents": contents})
        return [fp["fingerprint"] for fp in r["fingerprints"]]

    def hamming(self, a: str, b: str) -> dict:
        """Compute Hamming distance between two fingerprints (base64 or content)."""
        return self._post("/api/v1/hamming", {"a": a, "b": b})

    def similarity(self, a: str, b: str) -> float:
        """Compute similarity between two fingerprints."""
        r = self.hamming(a, b)
        return r["similarity"]

    def bind(self, a: str, b: str) -> str:
        """XOR bind two fingerprints."""
        r = self._post("/api/v1/bind", {"a": a, "b": b})
        return r["result"]

    def bundle(self, fingerprints: List[str]) -> str:
        """Majority-vote bundle of multiple fingerprints."""
        r = self._post("/api/v1/bundle", {"fingerprints": fingerprints})
        return r["result"]

    # --- Search ---
    def topk(self, query: str, k: int = 10) -> List[dict]:
        """Top-k search by Hamming distance."""
        r = self._post("/api/v1/search/topk", {"query": query, "k": k})
        return r["results"]

    def threshold(self, query: str, max_distance: int = 2000, limit: int = 100) -> List[dict]:
        """Threshold search — all within Hamming distance."""
        r = self._post("/api/v1/search/threshold", {
            "query": query, "max_distance": max_distance, "limit": limit
        })
        return r["results"]

    def resonate(self, content: str, threshold: float = 0.7, limit: int = 10) -> List[dict]:
        """Content-based resonance search."""
        r = self._post("/api/v1/search/resonate", {
            "content": content, "threshold": threshold, "limit": limit
        })
        return r["results"]

    # --- Index ---
    def index(self, content: str = None, fingerprint: str = None,
              id: str = None, metadata: dict = None) -> dict:
        """Add a fingerprint to the index."""
        data = {}
        if id: data["id"] = id
        if content: data["content"] = content
        if fingerprint: data["fingerprint"] = fingerprint
        if metadata: data["metadata"] = metadata
        return self._post("/api/v1/index", data)

    def index_count(self) -> int:
        return self._get("/api/v1/index/count")["count"]

    def index_clear(self) -> dict:
        return self._session.delete(f"{self.url}/api/v1/index", timeout=self.timeout).json()

    # --- NARS Inference ---
    def deduction(self, f1: float, c1: float, f2: float, c2: float) -> dict:
        return self._post("/api/v1/nars/deduction", {"f1": f1, "c1": c1, "f2": f2, "c2": c2})

    def induction(self, f1: float, c1: float, f2: float, c2: float) -> dict:
        return self._post("/api/v1/nars/induction", {"f1": f1, "c1": c1, "f2": f2, "c2": c2})

    def abduction(self, f1: float, c1: float, f2: float, c2: float) -> dict:
        return self._post("/api/v1/nars/abduction", {"f1": f1, "c1": c1, "f2": f2, "c2": c2})

    def revision(self, f1: float, c1: float, f2: float, c2: float) -> dict:
        return self._post("/api/v1/nars/revision", {"f1": f1, "c1": c1, "f2": f2, "c2": c2})

    # --- SQL / Cypher ---
    def sql(self, query: str) -> dict:
        return self._post("/api/v1/sql", {"query": query})

    def cypher(self, query: str) -> dict:
        return self._post("/api/v1/cypher", {"query": query})

    # --- Redis protocol ---
    def redis(self, command: str) -> dict:
        r = self._session.post(f"{self.url}/redis",
                               data=command.encode(),
                               headers={"Content-Type": "text/plain"},
                               timeout=self.timeout)
        return r.json()


# =============================================================================
# LANCEDB-COMPATIBLE API
# =============================================================================

class LadybugTable:
    """LanceDB-compatible table interface backed by LadybugDB."""

    def __init__(self, client: Client, name: str, data: List[dict] = None):
        self.client = client
        self.name = name
        if data:
            for row in data:
                text = row.get("text", row.get("content", str(row)))
                meta = {k: str(v) for k, v in row.items() if k not in ("text", "content", "vector")}
                self.client.index(content=text, id=row.get("id"), metadata=meta)

    def add(self, data: List[dict]):
        """Add rows to the table."""
        for row in data:
            text = row.get("text", row.get("content", str(row)))
            meta = {k: str(v) for k, v in row.items() if k not in ("text", "content", "vector")}
            self.client.index(content=text, id=row.get("id"), metadata=meta)

    def search(self, query: str = None, vector: list = None) -> "SearchBuilder":
        """Start a search query (LanceDB-compatible)."""
        return SearchBuilder(self.client, query or "", self.name)

    def count_rows(self) -> int:
        return self.client.index_count()

    def __len__(self) -> int:
        return self.count_rows()


class SearchBuilder:
    """LanceDB-compatible search builder with method chaining."""

    def __init__(self, client: Client, query: str, table: str):
        self._client = client
        self._query = query
        self._table = table
        self._limit = 10
        self._threshold = None
        self._metric = "hamming"

    def limit(self, n: int) -> "SearchBuilder":
        self._limit = n
        return self

    def metric(self, m: str) -> "SearchBuilder":
        self._metric = m
        return self

    def where(self, condition: str) -> "SearchBuilder":
        # Future: filter conditions
        return self

    def to_list(self) -> List[dict]:
        """Execute search and return results."""
        if self._threshold is not None:
            return self._client.threshold(self._query, max_distance=self._threshold, limit=self._limit)
        return self._client.topk(self._query, k=self._limit)

    def to_pandas(self):
        """Convert results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.to_list())


class LadybugDB:
    """LanceDB-compatible database connection."""

    def __init__(self, uri: str = "http://localhost:8080"):
        self.uri = uri
        self.client = Client(uri)
        self._tables: Dict[str, LadybugTable] = {}

    def create_table(self, name: str, data: List[dict] = None, **kwargs) -> LadybugTable:
        """Create a new table (LanceDB-compatible)."""
        table = LadybugTable(self.client, name, data)
        self._tables[name] = table
        return table

    def open_table(self, name: str) -> LadybugTable:
        """Open an existing table."""
        if name in self._tables:
            return self._tables[name]
        table = LadybugTable(self.client, name)
        self._tables[name] = table
        return table

    def table_names(self) -> List[str]:
        return list(self._tables.keys())

    def drop_table(self, name: str):
        self._tables.pop(name, None)
        self.client.index_clear()


def connect(uri: str = "http://localhost:8080") -> LadybugDB:
    """Connect to LadybugDB (LanceDB-compatible entry point)."""
    return LadybugDB(uri)


# =============================================================================
# NATIVE BINDINGS WRAPPER (if compiled with PyO3)
# =============================================================================

class NativeClient:
    """Wrapper around native PyO3 bindings for maximum performance."""

    def __init__(self):
        if not HAS_NATIVE:
            raise ImportError("Native ladybug module not found. Compile with: maturin develop --features python")
        self._native = _native

    def fingerprint(self, content: str) -> "NativeFingerprint":
        return NativeFingerprint(self._native.Fingerprint(content))

    def fingerprint_random(self) -> "NativeFingerprint":
        return NativeFingerprint(self._native.Fingerprint.random())

    def hamming(self, a: bytes, b: bytes) -> int:
        return self._native.hamming_bytes(a, b)

    def batch_hamming(self, query: "NativeFingerprint", candidates: list) -> list:
        return self._native.batch_hamming(query._fp, [c._fp for c in candidates])

    def topk(self, query: "NativeFingerprint", candidates: list, k: int = 10) -> list:
        return self._native.topk_hamming(query._fp, [c._fp for c in candidates], k)

    def bundle(self, fingerprints: list) -> "NativeFingerprint":
        return NativeFingerprint(self._native.bundle([fp._fp for fp in fingerprints]))

    def simd_level(self) -> str:
        return self._native.simd_level()

    def open_db(self, path: str) -> Any:
        return self._native.open(path)


class NativeFingerprint:
    """Wrapper around native Fingerprint."""

    def __init__(self, fp):
        self._fp = fp

    @classmethod
    def from_content(cls, content: str) -> "NativeFingerprint":
        return cls(_native.Fingerprint(content))

    @classmethod
    def from_bytes(cls, data: bytes) -> "NativeFingerprint":
        return cls(_native.Fingerprint.from_bytes(data))

    @classmethod
    def random(cls) -> "NativeFingerprint":
        return cls(_native.Fingerprint.random())

    def to_bytes(self) -> bytes:
        return self._fp.to_bytes()

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode()

    def hamming(self, other: "NativeFingerprint") -> int:
        return self._fp.hamming(other._fp)

    def similarity(self, other: "NativeFingerprint") -> float:
        return self._fp.similarity(other._fp)

    def bind(self, other: "NativeFingerprint") -> "NativeFingerprint":
        return NativeFingerprint(self._fp.bind(other._fp))

    def popcount(self) -> int:
        return self._fp.popcount()

    def density(self) -> float:
        return self._fp.density()

    def __repr__(self) -> str:
        return f"Fingerprint(popcount={self.popcount()}, density={self.density():.3f})"


# =============================================================================
# CONVENIENCE: auto-select best backend
# =============================================================================

def auto_client(url: str = None) -> Union[Client, NativeClient]:
    """Auto-select the best available client.

    - If url is provided → HTTP Client
    - If native bindings available → NativeClient
    - Otherwise → HTTP Client on localhost:8080
    """
    if url:
        return Client(url)
    if HAS_NATIVE:
        return NativeClient()
    return Client("http://localhost:8080")
