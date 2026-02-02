# Ladybug-RS Documentation

**Ladybug-RS** is a pure-Rust cognitive substrate implementing the 8+8 address model with Redis syntax and cognitive semantics.

## Documentation Structure

```
docs/
├── README.md                    # This file
├── architecture/
│   ├── OVERVIEW.md              # System architecture overview
│   ├── ADDRESS_MODEL.md         # 8+8 addressing (prefix:slot)
│   ├── MEMORY_ZONES.md          # Surface/Fluid/Node zones
│   └── DATA_FLOW.md             # Request/response flow
├── modules/
│   ├── STORAGE.md               # BindSpace, CogRedis, Lance
│   ├── SEARCH.md                # HDR Cascade, Cognitive search
│   ├── FLIGHT.md                # Arrow Flight server
│   ├── LEARNING.md              # CAM ops, Quantum ops, RL
│   └── COGNITIVE.md             # Grammar engine, Substrate
├── api/
│   ├── REDIS_COMMANDS.md        # Cognitive Redis command reference
│   ├── FLIGHT_ENDPOINTS.md      # Arrow Flight API
│   └── MCP_ACTIONS.md           # MCP tool actions
└── guides/
    ├── GETTING_STARTED.md       # Quick start guide
    ├── DEPLOYMENT.md            # Production deployment
    └── DEVELOPMENT.md           # Contributing guide
```

## Quick Links

- [Architecture Overview](architecture/OVERVIEW.md)
- [Flight Server API](api/FLIGHT_ENDPOINTS.md)
- [Redis Command Reference](api/REDIS_COMMANDS.md)
- [Getting Started](guides/GETTING_STARTED.md)

## Key Concepts

### 8+8 Address Model

65,536 addresses via 16-bit addressing: `prefix:slot` (u8:u8)

| Zone | Prefix Range | Addresses | Purpose |
|------|-------------|-----------|---------|
| Surface | 0x00-0x0F | 4,096 | Query language mappings |
| Fluid | 0x10-0x7F | 28,672 | Edges, context, working memory |
| Nodes | 0x80-0xFF | 32,768 | Universal bind space |

### Core Components

1. **BindSpace** - Universal DTO with O(1) array indexing
2. **CogRedis** - Redis syntax with cognitive semantics
3. **HdrIndex** - HDR cascade search (~7ns per candidate)
4. **Arrow Flight** - Zero-copy streaming for MCP integration

## Version

- **Current**: v0.3.0
- **DataFusion**: 51 (DF 52 upgrade path documented)
- **Arrow**: 48.x / arrow-flight 57.2
- **Rust**: 1.70+
