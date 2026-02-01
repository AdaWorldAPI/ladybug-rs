#!/usr/bin/env python3
"""
Ladybug-RS Python Client SDK

A Python client for interacting with Ladybug-RS cognitive database server.
Compatible with LanceDB-style API for vector operations.

Installation:
    pip install requests

Usage:
    from ladybug_client import LadybugClient, LadybugVectorDB

    # Basic client
    client = LadybugClient("http://localhost:8080")

    # Health check
    print(client.health())

    # Create fingerprint
    fp = client.create_fingerprint("hello world")
    print(f"Fingerprint popcount: {fp['popcount']}")

    # LanceDB-compatible vector operations
    db = LadybugVectorDB("http://localhost:8080")
    db.insert([
        {"content": "The quick brown fox"},
        {"content": "jumps over the lazy dog"}
    ])
    results = db.search("brown fox", k=5)

Author: Ada Consciousness Project
License: Apache-2.0
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import urllib.request
import urllib.error


@dataclass
class SearchResult:
    """Result from vector similarity search"""
    addr: str
    distance: float
    similarity: float
    metadata: Optional[Dict[str, Any]] = None


class LadybugClient:
    """
    Low-level client for Ladybug-RS HTTP API.

    Provides direct access to all server endpoints.
    """

    def __init__(self, base_url: str = "http://localhost:8080", timeout: int = 30):
        """
        Initialize client.

        Args:
            base_url: Server URL (default: http://localhost:8080)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to server."""
        url = f"{self.base_url}{path}"
        headers = {"Content-Type": "application/json"}

        body = None
        if data is not None:
            body = json.dumps(data).encode('utf-8')

        req = urllib.request.Request(url, data=body, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            try:
                return json.loads(error_body)
            except:
                return {"error": str(e), "status": e.code}
        except urllib.error.URLError as e:
            return {"error": f"Connection failed: {e.reason}"}

    def health(self) -> Dict:
        """Check server health."""
        return self._request("GET", "/health")

    def info(self) -> Dict:
        """Get server information."""
        return self._request("GET", "/info")

    def create_fingerprint(self, content: str) -> Dict:
        """
        Create a fingerprint from content string.

        Args:
            content: Text content to fingerprint

        Returns:
            Dict with fingerprint info (popcount, density, hex_preview)
        """
        result = self._request("POST", "/fingerprint", {"content": content})
        if result.get("success"):
            return result.get("fingerprint", {})
        return result

    def redis_command(self, command: str) -> Dict:
        """
        Execute a Redis-like command.

        Supported commands:
            SET <content> [confidence]  - Store fingerprint
            GET <addr>                  - Retrieve fingerprint
            SCAN <cursor> [MATCH pattern] [COUNT n]
            RESONATE <content> <k>      - Similarity search
            CAM <operation> [args...]   - CAM operation

        Args:
            command: Redis-like command string

        Returns:
            Command result
        """
        return self._request("POST", "/redis", {"command": command})

    def sql_query(self, query: str) -> Dict:
        """
        Execute SQL query.

        Args:
            query: SQL query string

        Returns:
            Query results
        """
        return self._request("POST", "/sql", {"query": query})

    def cypher_query(self, query: str) -> Dict:
        """
        Execute Cypher graph query.

        Args:
            query: Cypher query string

        Returns:
            Query results with nodes and relationships
        """
        return self._request("POST", "/cypher", {"query": query})

    def cam_operation(self, operation: str, args: Optional[List[str]] = None) -> Dict:
        """
        Execute a CAM (Content-Addressable Memory) operation.

        Args:
            operation: Operation name (e.g., "BIND", "UNBIND", "RESONATE")
            args: List of content strings as arguments

        Returns:
            Operation result
        """
        return self._request("POST", f"/cam/{operation}", {"args": args or []})

    def vector_search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query content string
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1)

        Returns:
            List of SearchResult objects
        """
        result = self._request("POST", "/vectors/search", {
            "query": query,
            "k": k,
            "threshold": threshold
        })

        if not result.get("success"):
            return []

        return [
            SearchResult(
                addr=m.get("addr", ""),
                distance=m.get("distance", 0),
                similarity=m.get("similarity", 0)
            )
            for m in result.get("matches", [])
        ]

    def vector_insert(self, vectors: List[Dict[str, Any]]) -> Dict:
        """
        Insert vectors into the database.

        Args:
            vectors: List of dicts with 'content' or 'vector' key

        Returns:
            Insert result with addresses
        """
        return self._request("POST", "/vectors/insert", {"vectors": vectors})


class LadybugVectorDB:
    """
    LanceDB-compatible vector database interface.

    Provides a familiar API for users of LanceDB, Chroma, or Pinecone.
    """

    def __init__(self, uri: str = "http://localhost:8080"):
        """
        Connect to Ladybug-RS vector database.

        Args:
            uri: Server URI (http://host:port)
        """
        self.client = LadybugClient(uri)
        self._tables: Dict[str, 'LadybugTable'] = {}

    def create_table(
        self,
        name: str,
        data: Optional[List[Dict]] = None,
        **kwargs
    ) -> 'LadybugTable':
        """
        Create a new table (namespace).

        In Ladybug-RS, tables are logical groupings stored in the
        same address space with prefixed keys.

        Args:
            name: Table name
            data: Optional initial data to insert

        Returns:
            LadybugTable instance
        """
        table = LadybugTable(self.client, name)
        if data:
            table.add(data)
        self._tables[name] = table
        return table

    def open_table(self, name: str) -> 'LadybugTable':
        """Open an existing table."""
        if name not in self._tables:
            self._tables[name] = LadybugTable(self.client, name)
        return self._tables[name]

    def drop_table(self, name: str):
        """Drop a table."""
        if name in self._tables:
            del self._tables[name]

    def table_names(self) -> List[str]:
        """List all table names."""
        return list(self._tables.keys())

    # Convenience methods for single-table usage
    def insert(self, data: List[Dict[str, Any]]) -> Dict:
        """Insert vectors (uses default table)."""
        return self.client.vector_insert(data)

    def search(
        self,
        query: str,
        k: int = 10,
        threshold: float = 0.5
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        return self.client.vector_search(query, k, threshold)


class LadybugTable:
    """
    A table (namespace) in the vector database.

    Compatible with LanceDB Table interface.
    """

    def __init__(self, client: LadybugClient, name: str):
        self.client = client
        self.name = name
        self._count = 0

    def add(self, data: List[Dict[str, Any]]) -> int:
        """
        Add vectors to the table.

        Args:
            data: List of dicts with 'content' or 'vector' key

        Returns:
            Number of rows added
        """
        # Prefix content with table name for namespace isolation
        prefixed_data = []
        for row in data:
            if "content" in row:
                prefixed_data.append({
                    "content": f"{self.name}:{row['content']}"
                })
            else:
                prefixed_data.append(row)

        result = self.client.vector_insert(prefixed_data)
        added = result.get("inserted", 0)
        self._count += added
        return added

    def search(
        self,
        query: Union[str, List[float]],
        k: int = 10,
        threshold: float = 0.5,
        **kwargs
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query: Query string or vector
            k: Number of results
            threshold: Minimum similarity

        Returns:
            List of SearchResult
        """
        query_str = query if isinstance(query, str) else str(query)
        return self.client.vector_search(
            f"{self.name}:{query_str}",
            k=k,
            threshold=threshold
        )

    def count_rows(self) -> int:
        """Get approximate row count."""
        return self._count

    def delete(self, where: str) -> int:
        """Delete matching rows (placeholder)."""
        # Would need DELETE support in server
        return 0

    def update(self, updates: Dict[str, Any], where: str) -> int:
        """Update matching rows (placeholder)."""
        return 0


class CognitiveGraph:
    """
    Graph database interface using Cypher queries.

    Compatible with Neo4j Python driver patterns.
    """

    def __init__(self, uri: str = "http://localhost:8080"):
        self.client = LadybugClient(uri)

    def run(self, query: str, **parameters) -> Dict:
        """
        Run a Cypher query.

        Args:
            query: Cypher query string
            **parameters: Query parameters (substituted into query)

        Returns:
            Query result with nodes and relationships
        """
        # Simple parameter substitution
        for key, value in parameters.items():
            query = query.replace(f"${key}", repr(value))

        return self.client.cypher_query(query)

    def create_node(self, labels: List[str], properties: Dict) -> Dict:
        """Create a node with given labels and properties."""
        labels_str = ':'.join(labels)
        props_str = ', '.join(f"{k}: {repr(v)}" for k, v in properties.items())
        query = f"CREATE (n:{labels_str} {{{props_str}}}) RETURN n"
        return self.run(query)

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict] = None
    ) -> Dict:
        """Create a relationship between nodes."""
        props_str = ""
        if properties:
            props_str = ' {' + ', '.join(f"{k}: {repr(v)}" for k, v in properties.items()) + '}'

        query = f"""
        MATCH (a), (b)
        WHERE id(a) = {repr(from_id)} AND id(b) = {repr(to_id)}
        CREATE (a)-[r:{rel_type}{props_str}]->(b)
        RETURN r
        """
        return self.run(query)

    def match(self, pattern: str, where: Optional[str] = None) -> Dict:
        """Match nodes/relationships."""
        query = f"MATCH {pattern}"
        if where:
            query += f" WHERE {where}"
        query += " RETURN *"
        return self.run(query)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    import sys

    # Default to localhost, or use command line arg
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

    print(f"Connecting to Ladybug-RS at {host}")
    print("=" * 50)

    # Initialize clients
    client = LadybugClient(host)
    db = LadybugVectorDB(host)
    graph = CognitiveGraph(host)

    # 1. Health check
    print("\n1. Health Check:")
    health = client.health()
    print(f"   Status: {health}")

    # 2. Server info
    print("\n2. Server Info:")
    info = client.info()
    print(f"   Name: {info.get('name', 'unknown')}")
    print(f"   Version: {info.get('version', 'unknown')}")

    # 3. Create fingerprint
    print("\n3. Fingerprint Creation:")
    fp = client.create_fingerprint("Hello, Ladybug!")
    print(f"   Popcount: {fp.get('popcount', 'N/A')}")
    print(f"   Density: {fp.get('density', 'N/A')}")

    # 4. Vector operations (LanceDB-style)
    print("\n4. Vector Operations:")

    # Insert some vectors
    insert_result = db.insert([
        {"content": "The quick brown fox jumps over the lazy dog"},
        {"content": "A fast auburn fox leaps above a sleepy hound"},
        {"content": "Machine learning is transforming AI research"},
        {"content": "Deep neural networks enable powerful models"},
    ])
    print(f"   Inserted: {insert_result.get('inserted', 0)} vectors")

    # Search
    results = db.search("fox jumping", k=3)
    print(f"   Search for 'fox jumping' found {len(results)} results:")
    for r in results:
        print(f"      - Addr: {r.addr}, Similarity: {r.similarity:.4f}")

    # 5. Redis commands
    print("\n5. Redis Commands:")
    redis_result = client.redis_command("SCAN 0 COUNT 5")
    print(f"   SCAN result: {redis_result.get('message', 'N/A')}")

    # 6. CAM operations
    print("\n6. CAM Operations:")
    cam_result = client.cam_operation("BIND", ["concept_a", "concept_b"])
    print(f"   BIND result: {cam_result}")

    # 7. Cypher query
    print("\n7. Cypher Query:")
    cypher_result = graph.run("MATCH (n) RETURN n LIMIT 5")
    print(f"   Result: {cypher_result}")

    print("\n" + "=" * 50)
    print("Demo complete!")
