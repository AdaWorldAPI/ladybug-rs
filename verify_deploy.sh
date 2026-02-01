#!/usr/bin/env bash
# =============================================================================
# LadybugDB — Post-Deploy Verification
# =============================================================================
# Usage:
#   ./verify_deploy.sh                          # localhost:8080
#   ./verify_deploy.sh https://ladybugdb.up.railway.app
#   ./verify_deploy.sh http://ladybugdb.railway.internal:8080
# =============================================================================
set -euo pipefail

BASE="${1:-http://localhost:8080}"
PASS=0; FAIL=0; TOTAL=0

check() {
    local label="$1" method="$2" path="$3" body="${4:-}"
    TOTAL=$((TOTAL+1))
    
    if [ "$method" = "GET" ]; then
        resp=$(curl -sf -w "\n%{http_code}" "${BASE}${path}" 2>/dev/null) || { echo "  ✗ $label (connection failed)"; FAIL=$((FAIL+1)); return; }
    else
        resp=$(curl -sf -w "\n%{http_code}" -X POST -H "Content-Type: application/json" -d "$body" "${BASE}${path}" 2>/dev/null) || { echo "  ✗ $label (connection failed)"; FAIL=$((FAIL+1)); return; }
    fi
    
    code=$(echo "$resp" | tail -1)
    body_resp=$(echo "$resp" | head -n -1)
    
    if [ "$code" = "200" ]; then
        echo "  ✓ $label"
        PASS=$((PASS+1))
    else
        echo "  ✗ $label (HTTP $code)"
        FAIL=$((FAIL+1))
    fi
}

echo "╔═══════════════════════════════════════════════════════╗"
echo "║         LadybugDB Deployment Verification            ║"
echo "╠═══════════════════════════════════════════════════════╣"
echo "║  Target: ${BASE}"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

echo "── Health & Info ──────────────────────────────────────"
check "Health check"        GET  "/health"
check "Readiness check"     GET  "/ready"
check "Root info"           GET  "/"
check "Server info"         GET  "/api/v1/info"
check "SIMD capabilities"   GET  "/api/v1/simd"

echo ""
echo "── Fingerprint Operations ────────────────────────────"
check "Create from content"  POST "/api/v1/fingerprint"        '{"content":"hello world"}'
check "Create random"        POST "/api/v1/fingerprint"        '{}'
check "Batch create"         POST "/api/v1/fingerprint/batch"  '{"contents":["alpha","beta","gamma"]}'
check "Hamming distance"     POST "/api/v1/hamming"            '{"a":"hello","b":"world"}'
check "Similarity"           POST "/api/v1/similarity"         '{"a":"hello","b":"hello"}'

echo ""
echo "── VSA Operations ────────────────────────────────────"
check "BIND (XOR)"           POST "/api/v1/bind"               '{"a":"cat","b":"animal"}'
check "BUNDLE (majority)"    POST "/api/v1/bundle"             '{"fingerprints":["cat","dog","animal"]}'

echo ""
echo "── Index & Search ────────────────────────────────────"
check "Index content"        POST "/api/v1/index"              '{"id":"t1","content":"consciousness emerges from complexity"}'
check "Index content 2"      POST "/api/v1/index"              '{"id":"t2","content":"awareness is fundamental to being"}'
check "Index content 3"      POST "/api/v1/index"              '{"id":"t3","content":"the cat sat on the mat"}'
check "Index count"          GET  "/api/v1/index/count"
check "Top-K search"         POST "/api/v1/search/topk"        '{"query":"consciousness","k":3}'
check "Threshold search"     POST "/api/v1/search/threshold"   '{"query":"awareness","max_distance":4000}'
check "Resonance search"     POST "/api/v1/search/resonate"    '{"content":"complexity emerges","threshold":0.5,"limit":5}'

echo ""
echo "── NARS Inference ────────────────────────────────────"
check "Deduction"            POST "/api/v1/nars/deduction"     '{"f1":0.9,"c1":0.9,"f2":0.8,"c2":0.8}'
check "Induction"            POST "/api/v1/nars/induction"     '{"f1":0.9,"c1":0.9,"f2":0.8,"c2":0.8}'
check "Abduction"            POST "/api/v1/nars/abduction"     '{"f1":0.9,"c1":0.9,"f2":0.8,"c2":0.8}'
check "Revision"             POST "/api/v1/nars/revision"      '{"f1":0.9,"c1":0.9,"f2":0.7,"c2":0.6}'

echo ""
echo "── Multi-Interface ───────────────────────────────────"
check "SQL endpoint"         POST "/api/v1/sql"                '{"query":"SELECT * FROM thoughts"}'
check "Cypher endpoint"      POST "/api/v1/cypher"             '{"query":"MATCH (a)-[:CAUSES]->(b) RETURN b"}'
check "Redis PING"           POST "/redis"                     'PING'
check "Redis SET"            POST "/redis"                     'SET mykey myvalue'
check "Redis GET"            POST "/redis"                     'GET mykey'

echo ""
echo "── LanceDB-Compatible API ────────────────────────────"
check "Lance create table"   POST "/api/v1/lance/table"        '{"name":"thoughts"}'
check "Lance add"            POST "/api/v1/lance/add"          '{"id":"l1","text":"test lance entry"}'
check "Lance search"         POST "/api/v1/lance/search"       '{"query":"test","limit":5}'

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Results: ${PASS}/${TOTAL} passed, ${FAIL} failed"
echo "════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    echo "  ⚠  Some checks failed — review server logs"
    exit 1
else
    echo "  ✓  All checks passed — deployment is healthy"
    exit 0
fi
