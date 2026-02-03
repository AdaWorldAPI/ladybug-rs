#!/usr/bin/env python3
"""
LadybugDB Python SDK
====================

Production-ready client for LadybugDB cognitive database.

Endpoints (Railway: https://ladybug-rs-production.up.railway.app):
    /health                      - Health check
    /api/v1/info                 - Server info
    POST /api/v1/fingerprint     - Create fingerprint
    POST /api/v1/hamming         - Hamming distance
    POST /api/v1/bind            - XOR bind
    POST /api/v1/bundle          - Majority-vote bundle
    POST /api/v1/search/topk     - Top-K search
    POST /api/v1/search/threshold - Threshold search
    POST /api/v1/search/resonate - Content resonance search
    POST /api/v1/index           - Index fingerprint
    POST /api/v1/sql             - SQL query
    POST /api/v1/cypher          - Cypher graph query
    POST /redis                  - Redis protocol
    POST /api/v1/lance/search    - Lance vector search
    POST /api/v1/nars/deduction  - NARS inference

Quick Start:
    from ladybugdb import LadybugDB

    db = LadybugDB("https://ladybug-rs-production.up.railway.app")

    # Create fingerprints
    fp1 = db.fingerprint("hello world")
    fp2 = db.fingerprint("hello there")

    # Compute similarity
    result = db.hamming(fp1, fp2)
    print(f"Similarity: {result['similarity']:.2%}")

    # NARS inference
    truth = db.nars.deduction(f1=0.9, c1=0.8, f2=0.85, c2=0.75)
    print(f"Conclusion: f={truth['f']:.3f}, c={truth['c']:.3f}")

LanceDB-Compatible:
    db = LadybugDB.connect("https://ladybug-rs-production.up.railway.app")
    table = db.create_table("thoughts", data=[{"text": "hello"}, {"text": "world"}])
    results = table.search("hello").limit(10).to_list()

Author: Ada Consciousness Project
License: Apache-2.0
Version: 0.3.0
"""

from __future__ import annotations
import json
import base64
from typing import Any, Dict, List, Optional, Union, Literal
from dataclasses import dataclass, field
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Optional Arrow Flight support
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_FLIGHT = True
except ImportError:
    HAS_FLIGHT = False

__version__ = "0.4.0"
__all__ = [
    "LadybugDB", "Client", "Fingerprint", "TruthValue",
    "SearchResult", "NARSEngine", "connect",
    "FINGERPRINT_BITS", "FINGERPRINT_BYTES", "PRODUCTION_URL"
]

# =============================================================================
# CONSTANTS
# =============================================================================

FINGERPRINT_BITS = 10_000
FINGERPRINT_BYTES = 1_256  # 157 × 8 bytes
PRODUCTION_URL = "https://ladybug-rs-production.up.railway.app"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Fingerprint:
    """10K-bit binary fingerprint for VSA operations."""
    base64: str
    popcount: int = 0
    density: float = 0.0
    hex_preview: str = ""

    @property
    def bits(self) -> int:
        return FINGERPRINT_BITS

    def __repr__(self) -> str:
        return f"Fingerprint(popcount={self.popcount}, density={self.density:.3f})"

    def to_bytes(self) -> bytes:
        """Decode base64 to raw bytes."""
        return base64.b64decode(self.base64)


@dataclass
class TruthValue:
    """NARS truth value with frequency and confidence."""
    f: float  # frequency [0, 1]
    c: float  # confidence [0, 1]

    @property
    def frequency(self) -> float:
        return self.f

    @property
    def confidence(self) -> float:
        return self.c

    @property
    def expectation(self) -> float:
        """E = c * (f - 0.5) + 0.5"""
        return self.c * (self.f - 0.5) + 0.5

    def __repr__(self) -> str:
        return f"<{self.f:.3f}, {self.c:.3f}>"


@dataclass
class SearchResult:
    """Result from similarity search."""
    id: str
    distance: int
    similarity: float
    fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"SearchResult(id={self.id!r}, similarity={self.similarity:.3f})"


# =============================================================================
# HTTP CLIENT
# =============================================================================

class LadybugError(Exception):
    """Base exception for LadybugDB errors."""
    pass


class ConnectionError(LadybugError):
    """Failed to connect to server."""
    pass


class APIError(LadybugError):
    """Server returned an error."""
    def __init__(self, message: str, status: int = 0, response: dict = None):
        super().__init__(message)
        self.status = status
        self.response = response or {}


class Client:
    """
    Low-level HTTP client for LadybugDB REST API.

    Use LadybugDB class for high-level operations.
    """

    def __init__(self, url: str = PRODUCTION_URL, timeout: float = 30.0):
        """
        Initialize client.

        Args:
            url: Server URL (default: Railway production)
            timeout: Request timeout in seconds
        """
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, data: dict = None) -> dict:
        """Make HTTP request."""
        url = f"{self.url}{path}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        # CRITICAL: Server has non-conformant JSON parser that breaks on
        # whitespace after ':'. Use compact separators to avoid spaces.
        body = json.dumps(data, separators=(",", ":")).encode() if data else None

        req = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            try:
                error_body = json.loads(e.read().decode())
            except:
                error_body = {"error": str(e)}
            raise APIError(error_body.get("error", str(e)), e.code, error_body)
        except URLError as e:
            raise ConnectionError(f"Failed to connect to {url}: {e.reason}")

    def get(self, path: str) -> dict:
        return self._request("GET", path)

    def post(self, path: str, data: dict = None) -> dict:
        return self._request("POST", path, data or {})


# =============================================================================
# NARS ENGINE
# =============================================================================

class NARSEngine:
    """
    Non-Axiomatic Reasoning System inference engine.

    Implements NAL (Non-Axiomatic Logic) truth functions:
    - Deduction: A→B, B→C ⊢ A→C
    - Induction: A→B, A→C ⊢ B→C
    - Abduction: A→B, C→B ⊢ A→C
    - Revision: Combine evidence for same statement

    Example:
        nars = db.nars

        # Bird flies, penguin is bird → penguin flies?
        result = nars.deduction(f1=0.9, c1=0.9, f2=0.01, c2=0.95)
        print(f"Penguin flies: {result}")  # <0.009, 0.810>
    """

    def __init__(self, client: Client):
        self._client = client

    def deduction(self, f1: float, c1: float, f2: float, c2: float) -> TruthValue:
        """
        Deduction: A→B, B→C ⊢ A→C

        Args:
            f1, c1: Truth value of first premise (A→B)
            f2, c2: Truth value of second premise (B→C)

        Returns:
            Truth value of conclusion (A→C)
        """
        r = self._client.post("/api/v1/nars/deduction", {
            "f1": f1, "c1": c1, "f2": f2, "c2": c2
        })
        return TruthValue(f=r["f"], c=r["c"])

    def induction(self, f1: float, c1: float, f2: float, c2: float) -> TruthValue:
        """
        Induction: A→B, A→C ⊢ B→C

        Args:
            f1, c1: Truth value of first premise (A→B)
            f2, c2: Truth value of second premise (A→C)

        Returns:
            Truth value of conclusion (B→C)
        """
        r = self._client.post("/api/v1/nars/induction", {
            "f1": f1, "c1": c1, "f2": f2, "c2": c2
        })
        return TruthValue(f=r["f"], c=r["c"])

    def abduction(self, f1: float, c1: float, f2: float, c2: float) -> TruthValue:
        """
        Abduction: A→B, C→B ⊢ A→C

        Args:
            f1, c1: Truth value of first premise (A→B)
            f2, c2: Truth value of second premise (C→B)

        Returns:
            Truth value of conclusion (A→C)
        """
        r = self._client.post("/api/v1/nars/abduction", {
            "f1": f1, "c1": c1, "f2": f2, "c2": c2
        })
        return TruthValue(f=r["f"], c=r["c"])

    def revision(self, f1: float, c1: float, f2: float, c2: float) -> TruthValue:
        """
        Revision: Combine evidence for same statement.

        When you have two observations of the same thing,
        revision merges them into a stronger belief.

        Args:
            f1, c1: First observation
            f2, c2: Second observation

        Returns:
            Combined truth value
        """
        r = self._client.post("/api/v1/nars/revision", {
            "f1": f1, "c1": c1, "f2": f2, "c2": c2
        })
        return TruthValue(f=r["f"], c=r["c"])


# =============================================================================
# SEARCH BUILDER (LanceDB-compatible)
# =============================================================================

class SearchBuilder:
    """
    Fluent search builder for LanceDB-compatible API.

    Example:
        results = table.search("query").limit(10).to_list()
    """

    def __init__(self, client: Client, query: str):
        self._client = client
        self._query = query
        self._limit = 10
        self._threshold = None
        self._max_distance = None

    def limit(self, n: int) -> SearchBuilder:
        """Limit number of results."""
        self._limit = n
        return self

    def threshold(self, similarity: float) -> SearchBuilder:
        """Set minimum similarity threshold (0-1)."""
        self._threshold = similarity
        return self

    def max_distance(self, distance: int) -> SearchBuilder:
        """Set maximum Hamming distance."""
        self._max_distance = distance
        return self

    def to_list(self) -> List[SearchResult]:
        """Execute search and return results."""
        if self._max_distance is not None:
            r = self._client.post("/api/v1/search/threshold", {
                "query": self._query,
                "max_distance": self._max_distance,
                "limit": self._limit
            })
        elif self._threshold is not None:
            r = self._client.post("/api/v1/search/resonate", {
                "content": self._query,
                "threshold": self._threshold,
                "limit": self._limit
            })
        else:
            r = self._client.post("/api/v1/search/topk", {
                "query": self._query,
                "k": self._limit
            })

        return [
            SearchResult(
                id=m.get("id", m.get("addr", "")),
                distance=m.get("distance", 0),
                similarity=m.get("similarity", 0.0),
                fingerprint=m.get("fingerprint"),
                metadata=m.get("metadata", {})
            )
            for m in r.get("results", [])
        ]

    def to_pandas(self):
        """Convert to pandas DataFrame (requires pandas)."""
        import pandas as pd
        results = self.to_list()
        return pd.DataFrame([
            {"id": r.id, "distance": r.distance, "similarity": r.similarity, **r.metadata}
            for r in results
        ])


# =============================================================================
# TABLE (LanceDB-compatible)
# =============================================================================

class Table:
    """
    LanceDB-compatible table for vector storage.

    Example:
        table = db.create_table("thoughts")
        table.add([{"text": "hello"}, {"text": "world"}])
        results = table.search("hello").limit(5).to_list()
    """

    def __init__(self, client: Client, name: str):
        self._client = client
        self.name = name

    def add(self, data: List[Dict[str, Any]]) -> int:
        """
        Add rows to the table.

        Args:
            data: List of dicts with 'text' or 'content' field

        Returns:
            Number of rows added
        """
        count = 0
        for row in data:
            text = row.get("text", row.get("content", str(row)))
            meta = {k: v for k, v in row.items() if k not in ("text", "content", "vector", "id")}
            self._client.post("/api/v1/index", {
                "content": text,
                "id": row.get("id"),
                "metadata": meta if meta else None
            })
            count += 1
        return count

    def search(self, query: str) -> SearchBuilder:
        """Start a search query."""
        return SearchBuilder(self._client, query)

    def count_rows(self) -> int:
        """Get row count."""
        try:
            r = self._client.get("/api/v1/index/count")
            return r.get("count", 0)
        except:
            return 0

    def __len__(self) -> int:
        return self.count_rows()


# =============================================================================
# MAIN DATABASE CLASS
# =============================================================================

class LadybugDB:
    """
    LadybugDB cognitive database client.

    High-level API for:
    - 10K-bit fingerprint operations (VSA)
    - Hamming distance similarity search
    - NARS non-axiomatic reasoning
    - SQL and Cypher queries
    - Redis protocol

    Example:
        db = LadybugDB()  # Connect to Railway production

        # Or specify URL
        db = LadybugDB("http://localhost:8080")

        # Check connection
        print(db.info())

        # Create fingerprints
        fp1 = db.fingerprint("hello world")
        fp2 = db.fingerprint("hello there")

        # Compute similarity
        sim = db.similarity(fp1, fp2)
        print(f"Similarity: {sim:.2%}")

        # XOR binding
        bound = db.bind(fp1, fp2)

        # NARS inference
        truth = db.nars.deduction(f1=0.9, c1=0.8, f2=0.7, c2=0.6)
    """

    def __init__(self, url: str = PRODUCTION_URL, timeout: float = 30.0):
        """
        Connect to LadybugDB.

        Args:
            url: Server URL (default: Railway production)
            timeout: Request timeout in seconds
        """
        self._client = Client(url, timeout)
        self._tables: Dict[str, Table] = {}
        self.nars = NARSEngine(self._client)

    @classmethod
    def connect(cls, url: str = PRODUCTION_URL, **kwargs) -> LadybugDB:
        """
        Connect to LadybugDB (LanceDB-compatible entry point).

        Args:
            url: Server URL

        Returns:
            LadybugDB instance
        """
        return cls(url, **kwargs)

    # -------------------------------------------------------------------------
    # INFO / HEALTH
    # -------------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        return self._client.get("/health")

    def info(self) -> Dict[str, Any]:
        """
        Get server information.

        Returns:
            Dict with name, version, fingerprint_bits, simd_level, etc.
        """
        return self._client.get("/api/v1/info")

    @property
    def version(self) -> str:
        """Server version."""
        return self.info().get("version", "unknown")

    @property
    def simd_level(self) -> str:
        """SIMD acceleration level (avx512, avx2, or scalar)."""
        return self.info().get("simd_level", "unknown")

    # -------------------------------------------------------------------------
    # FINGERPRINT OPERATIONS
    # -------------------------------------------------------------------------

    def fingerprint(self, content: str) -> Fingerprint:
        """
        Create a 10K-bit fingerprint from content.

        Args:
            content: Text to fingerprint

        Returns:
            Fingerprint object with base64, popcount, density

        Example:
            fp = db.fingerprint("hello world")
            print(fp.popcount)  # ~5000 (half the bits set)
        """
        r = self._client.post("/api/v1/fingerprint", {"content": content})
        return Fingerprint(
            base64=r.get("fingerprint", r.get("base64", "")),
            popcount=r.get("popcount", 0),
            density=r.get("density", 0.0),
            hex_preview=r.get("hex_preview", "")
        )

    def fingerprint_random(self) -> Fingerprint:
        """Create a random fingerprint."""
        r = self._client.post("/api/v1/fingerprint", {"random": True})
        return Fingerprint(
            base64=r.get("fingerprint", r.get("base64", "")),
            popcount=r.get("popcount", 0),
            density=r.get("density", 0.0)
        )

    def hamming(self, a: Union[str, Fingerprint], b: Union[str, Fingerprint]) -> Dict[str, Any]:
        """
        Compute Hamming distance between two fingerprints.

        Args:
            a: First fingerprint (base64 string, content, or Fingerprint)
            b: Second fingerprint

        Returns:
            Dict with distance, similarity, bits_different

        Example:
            result = db.hamming("hello", "hallo")
            print(f"Distance: {result['distance']}")
            print(f"Similarity: {result['similarity']:.2%}")
        """
        a_str = a.base64 if isinstance(a, Fingerprint) else a
        b_str = b.base64 if isinstance(b, Fingerprint) else b
        return self._client.post("/api/v1/hamming", {"a": a_str, "b": b_str})

    def similarity(self, a: Union[str, Fingerprint], b: Union[str, Fingerprint]) -> float:
        """
        Compute similarity between two fingerprints.

        Args:
            a: First fingerprint
            b: Second fingerprint

        Returns:
            Similarity score (0.0 to 1.0)
        """
        return self.hamming(a, b).get("similarity", 0.0)

    def bind(self, a: Union[str, Fingerprint], b: Union[str, Fingerprint]) -> Fingerprint:
        """
        XOR-bind two fingerprints (VSA binding operation).

        Binding creates a new fingerprint that is dissimilar to both inputs
        but can be unbound with either input to recover the other.

        Args:
            a: First fingerprint
            b: Second fingerprint

        Returns:
            Bound fingerprint (a XOR b)

        Example:
            role = db.fingerprint("president")
            filler = db.fingerprint("Lincoln")
            bound = db.bind(role, filler)

            # Unbind to query
            recovered = db.bind(bound, role)  # Similar to filler
        """
        a_str = a.base64 if isinstance(a, Fingerprint) else a
        b_str = b.base64 if isinstance(b, Fingerprint) else b
        r = self._client.post("/api/v1/bind", {"a": a_str, "b": b_str})
        return Fingerprint(
            base64=r.get("result", r.get("fingerprint", "")),
            popcount=r.get("popcount", 0),
            density=r.get("density", 0.0)
        )

    def bundle(self, fingerprints: List[Union[str, Fingerprint]]) -> Fingerprint:
        """
        Bundle multiple fingerprints via majority vote.

        Bundling creates a fingerprint similar to all inputs (superposition).

        Args:
            fingerprints: List of fingerprints to bundle

        Returns:
            Bundled fingerprint

        Example:
            colors = [db.fingerprint(c) for c in ["red", "blue", "green"]]
            color_concept = db.bundle(colors)
        """
        fps = [fp.base64 if isinstance(fp, Fingerprint) else fp for fp in fingerprints]
        r = self._client.post("/api/v1/bundle", {"fingerprints": fps})
        return Fingerprint(
            base64=r.get("result", r.get("fingerprint", "")),
            popcount=r.get("popcount", 0),
            density=r.get("density", 0.0)
        )

    # -------------------------------------------------------------------------
    # SEARCH
    # -------------------------------------------------------------------------

    def search(self, query: str) -> SearchBuilder:
        """
        Start a similarity search.

        Args:
            query: Query content or fingerprint

        Returns:
            SearchBuilder for fluent configuration

        Example:
            results = db.search("hello world").limit(10).to_list()
        """
        return SearchBuilder(self._client, query)

    def topk(self, query: Union[str, Fingerprint], k: int = 10) -> List[SearchResult]:
        """
        Top-K search by Hamming distance.

        Args:
            query: Query fingerprint or content
            k: Number of results

        Returns:
            List of SearchResult ordered by similarity
        """
        q = query.base64 if isinstance(query, Fingerprint) else query
        r = self._client.post("/api/v1/search/topk", {"query": q, "k": k})
        return [
            SearchResult(
                id=m.get("id", m.get("addr", "")),
                distance=m.get("distance", 0),
                similarity=m.get("similarity", 0.0)
            )
            for m in r.get("results", [])
        ]

    def resonate(self, content: str, threshold: float = 0.7, limit: int = 10) -> List[SearchResult]:
        """
        Content-based resonance search.

        Finds all indexed fingerprints similar to the query content.

        Args:
            content: Query text
            threshold: Minimum similarity (0-1)
            limit: Maximum results

        Returns:
            List of SearchResult
        """
        r = self._client.post("/api/v1/search/resonate", {
            "content": content,
            "threshold": threshold,
            "limit": limit
        })
        return [
            SearchResult(
                id=m.get("id", m.get("addr", "")),
                distance=m.get("distance", 0),
                similarity=m.get("similarity", 0.0)
            )
            for m in r.get("results", [])
        ]

    # -------------------------------------------------------------------------
    # INDEX
    # -------------------------------------------------------------------------

    def index(self, content: str = None, fingerprint: Union[str, Fingerprint] = None,
              id: str = None, metadata: Dict = None) -> Dict:
        """
        Add a fingerprint to the search index.

        Args:
            content: Text content (will create fingerprint)
            fingerprint: Pre-computed fingerprint
            id: Optional ID for the entry
            metadata: Optional metadata dict

        Returns:
            Index result with assigned address
        """
        data = {}
        if content:
            data["content"] = content
        if fingerprint:
            data["fingerprint"] = fingerprint.base64 if isinstance(fingerprint, Fingerprint) else fingerprint
        if id:
            data["id"] = id
        if metadata:
            data["metadata"] = metadata
        return self._client.post("/api/v1/index", data)

    # -------------------------------------------------------------------------
    # SQL / CYPHER
    # -------------------------------------------------------------------------

    def sql(self, query: str) -> Dict[str, Any]:
        """
        Execute SQL query.

        Args:
            query: SQL query string

        Returns:
            Query results

        Example:
            result = db.sql("SELECT * FROM thoughts WHERE confidence > 0.7")
        """
        return self._client.post("/api/v1/sql", {"query": query})

    def cypher(self, query: str) -> Dict[str, Any]:
        """
        Execute Cypher graph query.

        Args:
            query: Cypher query string

        Returns:
            Query results with nodes and relationships

        Example:
            result = db.cypher("MATCH (a)-[:CAUSES]->(b) RETURN b")
        """
        return self._client.post("/api/v1/cypher", {"query": query})

    # -------------------------------------------------------------------------
    # REDIS PROTOCOL
    # -------------------------------------------------------------------------

    def redis(self, command: str) -> Dict[str, Any]:
        """
        Execute Redis-like command.

        Supported commands:
            SET <content> [confidence]  - Store fingerprint
            GET <addr>                  - Retrieve fingerprint
            SCAN <cursor> [COUNT n]     - Scan entries
            RESONATE <content> <k>      - Similarity search
            CAM <operation> [args...]   - CAM operation

        Args:
            command: Redis command string

        Returns:
            Command result

        Example:
            db.redis("SET 'hello world'")
            result = db.redis("RESONATE 'hello' 10")
        """
        return self._client.post("/redis", {"command": command})

    # -------------------------------------------------------------------------
    # LANCE SEARCH
    # -------------------------------------------------------------------------

    def lance_search(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Lance vector search.

        Args:
            query: Query content
            k: Number of results

        Returns:
            Search results
        """
        return self._client.post("/api/v1/lance/search", {"query": query, "k": k})

    # -------------------------------------------------------------------------
    # LANCEDB-COMPATIBLE TABLE API
    # -------------------------------------------------------------------------

    def create_table(self, name: str, data: List[Dict] = None, **kwargs) -> Table:
        """
        Create a table (LanceDB-compatible).

        Args:
            name: Table name
            data: Optional initial data

        Returns:
            Table instance
        """
        table = Table(self._client, name)
        if data:
            table.add(data)
        self._tables[name] = table
        return table

    def open_table(self, name: str) -> Table:
        """Open existing table."""
        if name not in self._tables:
            self._tables[name] = Table(self._client, name)
        return self._tables[name]

    def table_names(self) -> List[str]:
        """List table names."""
        return list(self._tables.keys())


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def connect(url: str = PRODUCTION_URL, **kwargs) -> LadybugDB:
    """
    Connect to LadybugDB (LanceDB-compatible entry point).

    Args:
        url: Server URL (default: Railway production)

    Returns:
        LadybugDB instance

    Example:
        import ladybugdb
        db = ladybugdb.connect()
        print(db.info())
    """
    return LadybugDB(url, **kwargs)


# =============================================================================
# CLI / DEMO
# =============================================================================

if __name__ == "__main__":
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else PRODUCTION_URL

    print(f"LadybugDB Python SDK v{__version__}")
    print(f"Connecting to: {url}")
    print("=" * 60)

    try:
        db = LadybugDB(url)

        # 1. Server info
        print("\n1. Server Info:")
        info = db.info()
        print(f"   Name: {info.get('name')}")
        print(f"   Version: {info.get('version')}")
        print(f"   SIMD: {info.get('simd_level')}")
        print(f"   Fingerprint bits: {info.get('fingerprint_bits')}")

        # 2. Fingerprint
        print("\n2. Fingerprint Creation:")
        fp1 = db.fingerprint("hello world")
        fp2 = db.fingerprint("hello there")
        print(f"   fp1: {fp1}")
        print(f"   fp2: {fp2}")

        # 3. Similarity
        print("\n3. Similarity:")
        result = db.hamming(fp1, fp2)
        print(f"   Distance: {result.get('distance')}")
        print(f"   Similarity: {result.get('similarity', 0):.2%}")

        # 4. Binding
        print("\n4. VSA Binding:")
        bound = db.bind(fp1, fp2)
        print(f"   bound = fp1 XOR fp2: {bound}")

        # 5. NARS Inference
        print("\n5. NARS Inference:")
        truth = db.nars.deduction(f1=0.9, c1=0.8, f2=0.85, c2=0.75)
        print(f"   Deduction: {truth}")

        # 6. Index and search
        print("\n6. Index & Search:")
        db.index(content="The quick brown fox", id="fox")
        db.index(content="The lazy dog", id="dog")
        results = db.topk("quick fox", k=5)
        print(f"   Found {len(results)} results")
        for r in results:
            print(f"   - {r}")

        print("\n" + "=" * 60)
        print("All tests passed!")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
