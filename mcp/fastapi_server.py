#!/usr/bin/env python3
"""
LadybugDB FastAPI Server

High-performance HTTP + WebSocket server with Arrow IPC zero-copy support.
Acts as a bridge between FastAPI and the Rust backend.

Usage:
    uvicorn fastapi_server:app --host 0.0.0.0 --port 8000

Features:
    - JSON endpoints (standard)
    - Arrow IPC endpoints (zero-copy via shared memory)
    - WebSocket streaming for real-time results
    - MCP tool compatibility
"""

import asyncio
import base64
import json
import mmap
import os
import struct
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import httpx

# Optional Arrow support for zero-copy
try:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    print("Warning: pyarrow not installed. Zero-copy disabled.")

# =============================================================================
# Configuration
# =============================================================================

RUST_BACKEND = os.getenv("LADYBUG_BACKEND", "http://localhost:8080")
SHARED_MEM_PATH = Path(os.getenv("LADYBUG_SHM_PATH", "/dev/shm/ladybug"))

# =============================================================================
# Pydantic Models
# =============================================================================

class EncodingStyle(str, Enum):
    creative = "creative"
    balanced = "balanced"
    precise = "precise"

class EncodeRequest(BaseModel):
    text: Optional[str] = None
    data: Optional[str] = None  # hex-encoded
    style: EncodingStyle = EncodingStyle.balanced

class EncodeResponse(BaseModel):
    fingerprint: str  # base64
    popcount: int
    density: float
    encoding_style: str

class BindRequest(BaseModel):
    address: int = Field(..., ge=0, le=65535)
    fingerprint: str  # base64
    label: Optional[str] = None

class ReadRequest(BaseModel):
    address: int = Field(..., ge=0, le=65535)

class SearchRequest(BaseModel):
    query: str  # base64 fingerprint or text
    k: int = Field(default=10, ge=1, le=1000)
    threshold: Optional[int] = None
    stream: bool = False

class HammingRequest(BaseModel):
    a: str
    b: str

class XorBindRequest(BaseModel):
    a: str
    b: str

class SearchResult(BaseModel):
    address: int
    fingerprint: str
    label: Optional[str]
    distance: int
    similarity: float
    cascade_level: Optional[int] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time_ns: Optional[int] = None
    cascade_stats: Optional[Dict[str, int]] = None

# =============================================================================
# Zero-Copy Infrastructure
# =============================================================================

@dataclass
class SharedMemoryRegion:
    """Shared memory region for zero-copy Arrow IPC."""
    name: str
    path: Path
    size: int
    mmap: Optional[mmap.mmap] = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        os.ftruncate(fd, self.size)
        self.mmap = mmap.mmap(fd, self.size)
        os.close(fd)
        return self

    def __exit__(self, *args):
        if self.mmap:
            self.mmap.close()
        if self.path.exists():
            self.path.unlink()

    def write_arrow(self, table: "pa.Table") -> int:
        """Write Arrow table to shared memory. Returns offset."""
        if not HAS_ARROW:
            raise RuntimeError("pyarrow not installed")

        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

        buf = sink.getvalue()
        offset = 0  # Could implement ring buffer here

        self.mmap.seek(offset)
        self.mmap.write(buf.to_pybytes())

        return offset

    def read_arrow(self, offset: int = 0) -> "pa.Table":
        """Read Arrow table from shared memory."""
        if not HAS_ARROW:
            raise RuntimeError("pyarrow not installed")

        self.mmap.seek(offset)
        # Read IPC stream from mmap
        reader = ipc.open_stream(pa.py_buffer(self.mmap))
        return reader.read_all()

class ZeroCopyCache:
    """
    Cache for zero-copy Arrow data sharing between Python FastAPI and Rust backend.

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Shared Memory (/dev/shm)                     │
    ├─────────────────────────────────────────────────────────────────┤
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
    │  │ Slot 0  │  │ Slot 1  │  │ Slot 2  │  │ Slot 3  │  ...       │
    │  │ Arrow   │  │ Arrow   │  │ Arrow   │  │ Arrow   │            │
    │  │ IPC     │  │ IPC     │  │ IPC     │  │ IPC     │            │
    │  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
    └─────────────────────────────────────────────────────────────────┘
                    ↑                           ↑
                    │                           │
            Python writes                Rust reads
            (producer)                  (consumer)
    ```

    Zero-copy is maintained by:
    1. Using mmap'd shared memory (no kernel copy)
    2. Arrow IPC format (self-describing, no parsing needed)
    3. Pointer-based slot references (no data movement)
    """

    def __init__(self, base_path: Path, num_slots: int = 16, slot_size: int = 16 * 1024 * 1024):
        self.base_path = base_path
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.regions: Dict[int, SharedMemoryRegion] = {}
        self.next_slot = 0

    def allocate_slot(self) -> SharedMemoryRegion:
        """Allocate next available slot."""
        slot_id = self.next_slot
        self.next_slot = (self.next_slot + 1) % self.num_slots

        if slot_id in self.regions:
            # Reuse existing slot
            return self.regions[slot_id]

        region = SharedMemoryRegion(
            name=f"slot_{slot_id}",
            path=self.base_path / f"slot_{slot_id}.arrow",
            size=self.slot_size,
        )
        region.__enter__()
        self.regions[slot_id] = region
        return region

    def cleanup(self):
        """Cleanup all shared memory regions."""
        for region in self.regions.values():
            region.__exit__(None, None, None)
        self.regions.clear()

# Global cache instance
zero_copy_cache: Optional[ZeroCopyCache] = None

# =============================================================================
# FastAPI Application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan for resource management."""
    global zero_copy_cache
    if HAS_ARROW:
        zero_copy_cache = ZeroCopyCache(SHARED_MEM_PATH)
    yield
    if zero_copy_cache:
        zero_copy_cache.cleanup()

app = FastAPI(
    title="LadybugDB API",
    description="Cognitive database with JSON and Arrow IPC endpoints",
    version="0.4.0",
    lifespan=lifespan,
)

# HTTP client for Rust backend
http_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup():
    global http_client
    http_client = httpx.AsyncClient(base_url=RUST_BACKEND, timeout=30.0)

@app.on_event("shutdown")
async def shutdown():
    if http_client:
        await http_client.aclose()

# =============================================================================
# JSON Endpoints (Standard)
# =============================================================================

@app.post("/api/v1/encode", response_model=EncodeResponse, tags=["Fingerprint"])
async def encode(request: EncodeRequest):
    """
    Encode text/data to 10K-bit fingerprint.

    Uses Sigma-10 membrane encoding for semantic fingerprinting.
    """
    resp = await http_client.post("/api/v1/fingerprint", json={
        "content": request.text,
        "data": request.data,
        "style": request.style.value,
    })
    resp.raise_for_status()
    data = resp.json()
    return EncodeResponse(
        fingerprint=data.get("fingerprint", data.get("base64", "")),
        popcount=data.get("popcount", 0),
        density=data.get("density", 0.0),
        encoding_style=request.style.value,
    )

@app.post("/api/v1/bind", tags=["BindSpace"])
async def bind(request: BindRequest):
    """Bind fingerprint to BindSpace address."""
    resp = await http_client.post("/api/v1/index", json={
        "address": request.address,
        "fingerprint": request.fingerprint,
        "label": request.label,
    })
    resp.raise_for_status()
    return resp.json()

@app.post("/api/v1/read", tags=["BindSpace"])
async def read(request: ReadRequest):
    """Read node from BindSpace address."""
    resp = await http_client.post("/api/v1/read", json={
        "address": request.address,
    })
    resp.raise_for_status()
    return resp.json()

@app.post("/api/v1/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search for similar fingerprints.

    Uses HDR cascade for efficient filtering:
    - L0: 1-bit sketch → 90% filtered
    - L1: 4-bit count → 90% of survivors
    - L2: 8-bit count → 90% of survivors
    - L3: Full Hamming → exact distance
    """
    resp = await http_client.post("/api/v1/search/topk", json={
        "query": request.query,
        "k": request.k,
        "threshold": request.threshold,
    })
    resp.raise_for_status()
    return resp.json()

@app.post("/api/v1/hamming", tags=["Operations"])
async def hamming(request: HammingRequest):
    """Compute Hamming distance between fingerprints."""
    resp = await http_client.post("/api/v1/hamming", json={
        "a": request.a,
        "b": request.b,
    })
    resp.raise_for_status()
    return resp.json()

@app.post("/api/v1/xor_bind", tags=["Operations"])
async def xor_bind(request: XorBindRequest):
    """XOR bind two fingerprints (holographic composition)."""
    resp = await http_client.post("/api/v1/bind", json={
        "a": request.a,
        "b": request.b,
    })
    resp.raise_for_status()
    return resp.json()

# =============================================================================
# Arrow IPC Endpoints (Zero-Copy)
# =============================================================================

@app.post("/api/v1/arrow/search", tags=["Arrow IPC"])
async def arrow_search(request: SearchRequest):
    """
    Search with Arrow IPC response (zero-copy).

    Returns Arrow IPC stream bytes that can be memory-mapped.

    Zero-copy path:
    1. Rust backend writes results to Arrow RecordBatch
    2. RecordBatch serialized to IPC format (in-place)
    3. Bytes returned without intermediate copies
    4. Client can mmap or use pa.ipc.open_stream()
    """
    if not HAS_ARROW:
        raise HTTPException(501, "Arrow IPC not available (pyarrow not installed)")

    # For now, get JSON and convert to Arrow
    # In production, Rust backend would return Arrow IPC directly
    resp = await http_client.post("/api/v1/search/topk", json={
        "query": request.query,
        "k": request.k,
    })
    resp.raise_for_status()
    data = resp.json()

    # Convert to Arrow table
    results = data.get("results", [])
    if not results:
        # Empty table with schema
        table = pa.table({
            "address": pa.array([], type=pa.uint16()),
            "distance": pa.array([], type=pa.uint32()),
            "similarity": pa.array([], type=pa.float32()),
        })
    else:
        table = pa.table({
            "address": pa.array([r.get("address", 0) for r in results], type=pa.uint16()),
            "distance": pa.array([r.get("distance", 0) for r in results], type=pa.uint32()),
            "similarity": pa.array([r.get("similarity", 0.0) for r in results], type=pa.float32()),
        })

    # Serialize to IPC
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)

    return Response(
        content=sink.getvalue().to_pybytes(),
        media_type="application/vnd.apache.arrow.stream",
    )

@app.post("/api/v1/arrow/ingest", tags=["Arrow IPC"])
async def arrow_ingest(request: bytes = ...):
    """
    Ingest fingerprints via Arrow IPC (zero-copy).

    Accepts Arrow IPC stream bytes containing fingerprints.
    """
    if not HAS_ARROW:
        raise HTTPException(501, "Arrow IPC not available")

    # Read Arrow table from request body
    reader = ipc.open_stream(pa.py_buffer(request))
    table = reader.read_all()

    # Forward to Rust backend (would use shared memory in production)
    count = 0
    for row in table.to_pylist():
        await http_client.post("/api/v1/index", json={
            "address": row.get("address"),
            "fingerprint": base64.b64encode(row.get("fingerprint", b"")).decode(),
            "label": row.get("label"),
        })
        count += 1

    return {"ingested": count}

@app.get("/api/v1/arrow/shm/{slot_id}", tags=["Arrow IPC"])
async def arrow_shm_read(slot_id: int):
    """
    Read Arrow data from shared memory slot (true zero-copy).

    Returns pointer to shared memory region for memory-mapped access.
    Client can use mmap to access data without any copies.
    """
    if not zero_copy_cache:
        raise HTTPException(501, "Zero-copy cache not available")

    if slot_id not in zero_copy_cache.regions:
        raise HTTPException(404, f"Slot {slot_id} not found")

    region = zero_copy_cache.regions[slot_id]
    return {
        "slot_id": slot_id,
        "path": str(region.path),
        "size": region.size,
        "hint": "mmap this path for zero-copy access",
    }

# =============================================================================
# WebSocket Streaming
# =============================================================================

@app.websocket("/ws/search")
async def websocket_search(websocket: WebSocket):
    """
    WebSocket endpoint for streaming search results.

    Protocol:
    1. Client sends: {"query": "base64...", "k": 100}
    2. Server streams: {"result": {...}} for each match
    3. Server sends: {"done": true, "stats": {...}}
    """
    await websocket.accept()

    try:
        while True:
            # Receive search request
            data = await websocket.receive_json()
            query = data.get("query")
            k = data.get("k", 10)

            if not query:
                await websocket.send_json({"error": "Missing query"})
                continue

            # Execute search and stream results
            resp = await http_client.post("/api/v1/search/topk", json={
                "query": query,
                "k": k,
            })
            results = resp.json().get("results", [])

            # Stream each result
            for i, result in enumerate(results):
                await websocket.send_json({
                    "result": result,
                    "index": i,
                    "total": len(results),
                })
                await asyncio.sleep(0.001)  # Small delay to prevent flooding

            # Send completion
            await websocket.send_json({
                "done": True,
                "total_results": len(results),
            })

    except WebSocketDisconnect:
        pass

# =============================================================================
# Health & Info
# =============================================================================

@app.get("/health")
async def health():
    return {"status": "ok", "arrow_support": HAS_ARROW}

@app.get("/api/v1/info")
async def info():
    resp = await http_client.get("/api/v1/info")
    data = resp.json()
    data["fastapi_version"] = "0.4.0"
    data["arrow_support"] = HAS_ARROW
    data["zero_copy_enabled"] = zero_copy_cache is not None
    return data

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
