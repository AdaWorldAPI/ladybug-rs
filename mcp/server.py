#!/usr/bin/env python3
"""
LadybugDB MCP Server (Python)

Model Context Protocol server with both JSON-RPC and Arrow Flight support.
Enables streaming search results and zero-copy fingerprint transfer.

Usage:
    python server.py --http http://localhost:8080 --flight grpc://localhost:50051

Environment:
    LADYBUG_HTTP_URL: HTTP endpoint (default: http://localhost:8080)
    LADYBUG_FLIGHT_URL: Flight endpoint (default: grpc://localhost:50051)
    LADYBUG_TRANSPORT: Transport type (json or flight, default: json)
"""

import asyncio
import json
import os
import sys
from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import HTTPError

# MCP SDK
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("Warning: mcp package not installed. Run: pip install mcp", file=sys.stderr)

# Arrow Flight (optional)
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_FLIGHT = True
except ImportError:
    HAS_FLIGHT = False

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    http_url: str = "http://localhost:8080"
    flight_url: str = "grpc://localhost:50051"
    transport: str = "json"  # "json" or "flight"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            http_url=os.getenv("LADYBUG_HTTP_URL", "http://localhost:8080"),
            flight_url=os.getenv("LADYBUG_FLIGHT_URL", "grpc://localhost:50051"),
            transport=os.getenv("LADYBUG_TRANSPORT", "json"),
        )

# =============================================================================
# HTTP Client (JSON transport)
# =============================================================================

class HTTPClient:
    """JSON-RPC client for LadybugDB."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def request(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP POST request."""
        url = f"{self.base_url}{path}"
        body = json.dumps(data, separators=(",", ":")).encode()
        req = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as e:
            error = json.loads(e.read().decode()) if e.fp else {"error": str(e)}
            raise Exception(f"HTTP {e.code}: {error.get('error', str(e))}")

# =============================================================================
# Flight Client (Arrow Flight transport)
# =============================================================================

class FlightClient:
    """Arrow Flight client for LadybugDB with streaming support."""

    def __init__(self, flight_url: str):
        if not HAS_FLIGHT:
            raise ImportError("pyarrow not installed. Run: pip install pyarrow")
        self.client = flight.connect(flight_url)

    def do_action(self, action_type: str, body: bytes) -> bytes:
        """Execute Flight action."""
        action = flight.Action(action_type, body)
        results = list(self.client.do_action(action))
        if results:
            return results[0].body.to_pybytes()
        return b"{}"

    def stream_search(
        self, query_hex: str, k: int = 10
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream search results as they're found."""
        ticket = flight.Ticket(f"topk:{query_hex}:{k}".encode())
        reader = self.client.do_get(ticket)

        for batch in reader:
            # Convert RecordBatch to dicts
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                yield row

    def stream_all(self) -> AsyncIterator[Dict[str, Any]]:
        """Stream all fingerprints."""
        ticket = flight.Ticket(b"all")
        reader = self.client.do_get(ticket)

        for batch in reader:
            table = pa.Table.from_batches([batch])
            for row in table.to_pylist():
                yield row

    def ingest(self, fingerprints: List[Dict[str, Any]]) -> int:
        """Ingest fingerprints via zero-copy transfer."""
        # Build Arrow table from fingerprints
        addresses = [fp["address"] for fp in fingerprints]
        fps = [fp["fingerprint"] for fp in fingerprints]
        labels = [fp.get("label", "") for fp in fingerprints]

        table = pa.table({
            "address": pa.array(addresses, type=pa.uint16()),
            "fingerprint": pa.array(fps, type=pa.binary()),
            "label": pa.array(labels, type=pa.string()),
        })

        # Upload via DoPut
        descriptor = flight.FlightDescriptor.for_path(["ingest"])
        writer, _ = self.client.do_put(descriptor, table.schema)
        writer.write_table(table)
        writer.close()

        return len(fingerprints)

# =============================================================================
# Tool Definitions
# =============================================================================

TOOLS = [
    Tool(
        name="ladybug_encode",
        description=(
            "Encode text or data to a 10K-bit fingerprint via Sigma-10 membrane encoding. "
            "Creates a binary representation suitable for similarity search and VSA operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to encode"},
                "data": {"type": "string", "description": "Hex-encoded binary data"},
                "style": {
                    "type": "string",
                    "enum": ["creative", "balanced", "precise"],
                    "description": "Encoding style",
                },
            },
        },
    ),
    Tool(
        name="ladybug_bind",
        description="Bind a fingerprint to a BindSpace address.",
        inputSchema={
            "type": "object",
            "properties": {
                "address": {"type": "integer", "description": "16-bit address"},
                "fingerprint": {"type": "string", "description": "Base64 fingerprint"},
                "label": {"type": "string", "description": "Optional label"},
            },
            "required": ["address", "fingerprint"],
        },
    ),
    Tool(
        name="ladybug_read",
        description="Read a node from BindSpace by address.",
        inputSchema={
            "type": "object",
            "properties": {
                "address": {"type": "integer", "description": "16-bit address"},
            },
            "required": ["address"],
        },
    ),
    Tool(
        name="ladybug_resonate",
        description=(
            "Find similar fingerprints using HDR cascade search. "
            "Returns results ordered by similarity."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Base64 query fingerprint"},
                "k": {"type": "integer", "description": "Number of results"},
                "threshold": {"type": "integer", "description": "Max Hamming distance"},
                "stream": {"type": "boolean", "description": "Stream results (Flight only)"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ladybug_hamming",
        description="Compute Hamming distance between two fingerprints.",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First fingerprint"},
                "b": {"type": "string", "description": "Second fingerprint"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="ladybug_xor_bind",
        description="XOR bind two fingerprints (holographic composition).",
        inputSchema={
            "type": "object",
            "properties": {
                "a": {"type": "string", "description": "First fingerprint"},
                "b": {"type": "string", "description": "Second fingerprint"},
            },
            "required": ["a", "b"],
        },
    ),
    Tool(
        name="ladybug_stats",
        description="Get BindSpace statistics.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="ladybug_sql",
        description="Execute SQL query with cognitive UDFs.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query"},
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="ladybug_cypher",
        description="Execute Cypher graph query.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Cypher query"},
            },
            "required": ["query"],
        },
    ),
]

# =============================================================================
# Tool Handlers
# =============================================================================

class LadybugMCP:
    """LadybugDB MCP tool handler."""

    def __init__(self, config: Config):
        self.config = config
        self.http = HTTPClient(config.http_url)
        self.flight: Optional[FlightClient] = None

        if config.transport == "flight" and HAS_FLIGHT:
            try:
                self.flight = FlightClient(config.flight_url)
            except Exception as e:
                print(f"Warning: Flight connection failed: {e}", file=sys.stderr)

    async def handle_tool(self, name: str, args: Dict[str, Any]) -> str:
        """Handle MCP tool call."""
        try:
            if name == "ladybug_encode":
                result = self.http.request("/api/v1/fingerprint", {
                    "content": args.get("text"),
                    "data": args.get("data"),
                    "style": args.get("style"),
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_bind":
                result = self.http.request("/api/v1/index", {
                    "address": args["address"],
                    "fingerprint": args["fingerprint"],
                    "label": args.get("label"),
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_read":
                result = self.http.request("/api/v1/read", {
                    "address": args["address"],
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_resonate":
                # Use Flight streaming if available and requested
                if args.get("stream") and self.flight:
                    results = []
                    query_hex = args["query"]  # Assume already hex
                    k = args.get("k", 10)
                    for row in self.flight.stream_search(query_hex, k):
                        results.append(row)
                        if len(results) >= k:
                            break
                    return json.dumps({"results": results, "transport": "flight"}, indent=2)
                else:
                    result = self.http.request("/api/v1/search/topk", {
                        "query": args["query"],
                        "k": args.get("k", 10),
                        "threshold": args.get("threshold"),
                    })
                    return json.dumps(result, indent=2)

            elif name == "ladybug_hamming":
                result = self.http.request("/api/v1/hamming", {
                    "a": args["a"],
                    "b": args["b"],
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_xor_bind":
                result = self.http.request("/api/v1/bind", {
                    "a": args["a"],
                    "b": args["b"],
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_stats":
                result = self.http.request("/api/v1/stats", {})
                return json.dumps(result, indent=2)

            elif name == "ladybug_sql":
                result = self.http.request("/api/v1/sql", {
                    "query": args["query"],
                })
                return json.dumps(result, indent=2)

            elif name == "ladybug_cypher":
                result = self.http.request("/api/v1/cypher", {
                    "query": args["query"],
                })
                return json.dumps(result, indent=2)

            else:
                return json.dumps({"error": f"Unknown tool: {name}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

# =============================================================================
# MCP Server
# =============================================================================

async def main():
    if not HAS_MCP:
        print("MCP SDK not installed. Run: pip install mcp", file=sys.stderr)
        sys.exit(1)

    config = Config.from_env()
    handler = LadybugMCP(config)

    server = Server("ladybug-mcp")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]):
        result = await handler.handle_tool(name, arguments)
        return [TextContent(type="text", text=result)]

    print(f"LadybugDB MCP server starting...", file=sys.stderr)
    print(f"  HTTP: {config.http_url}", file=sys.stderr)
    print(f"  Flight: {config.flight_url}", file=sys.stderr)
    print(f"  Transport: {config.transport}", file=sys.stderr)

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    asyncio.run(main())
