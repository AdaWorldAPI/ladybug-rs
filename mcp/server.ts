/**
 * LadybugDB MCP Server
 *
 * Model Context Protocol server for LadybugDB cognitive database.
 * Supports both JSON-RPC and Arrow Flight transports.
 *
 * Usage:
 *   npx ts-node server.ts --http http://localhost:8080 --flight grpc://localhost:50051
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
  TextContent,
} from "@modelcontextprotocol/sdk/types.js";

// =============================================================================
// Configuration
// =============================================================================

interface ServerConfig {
  httpUrl: string;
  flightUrl?: string;
  transport: "json" | "flight";
}

const DEFAULT_CONFIG: ServerConfig = {
  httpUrl: process.env.LADYBUG_HTTP_URL || "http://localhost:8080",
  flightUrl: process.env.LADYBUG_FLIGHT_URL || "grpc://localhost:50051",
  transport: (process.env.LADYBUG_TRANSPORT as "json" | "flight") || "json",
};

// =============================================================================
// Tool Definitions
// =============================================================================

const TOOLS: Tool[] = [
  {
    name: "ladybug_encode",
    description:
      "Encode text or data to a 10K-bit fingerprint via Sigma-10 membrane encoding. " +
      "Creates a binary representation suitable for similarity search and VSA operations.",
    inputSchema: {
      type: "object",
      properties: {
        text: {
          type: "string",
          description: "Text to encode into fingerprint",
        },
        data: {
          type: "string",
          description: "Hex-encoded binary data to encode",
        },
        style: {
          type: "string",
          enum: ["creative", "balanced", "precise"],
          description: "Encoding style affecting bit distribution",
        },
      },
    },
  },
  {
    name: "ladybug_bind",
    description:
      "Bind a fingerprint to a BindSpace address. " +
      "Addresses are 16-bit: 0x00-0x0F=surface, 0x10-0x7F=fluid, 0x80-0xFF=nodes.",
    inputSchema: {
      type: "object",
      properties: {
        address: {
          type: "integer",
          description: "16-bit address (0-65535)",
        },
        fingerprint: {
          type: "string",
          description: "Base64-encoded fingerprint",
        },
        label: {
          type: "string",
          description: "Optional label for the node",
        },
      },
      required: ["address", "fingerprint"],
    },
  },
  {
    name: "ladybug_read",
    description: "Read a node from BindSpace by address.",
    inputSchema: {
      type: "object",
      properties: {
        address: {
          type: "integer",
          description: "16-bit address to read",
        },
      },
      required: ["address"],
    },
  },
  {
    name: "ladybug_resonate",
    description:
      "Find similar fingerprints using HDR cascade search. " +
      "Returns results ordered by similarity with cascade statistics.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Base64-encoded query fingerprint",
        },
        k: {
          type: "integer",
          description: "Number of results (default: 10)",
        },
        threshold: {
          type: "integer",
          description: "Maximum Hamming distance threshold",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "ladybug_hamming",
    description:
      "Compute Hamming distance between two fingerprints. " +
      "Returns distance, similarity (0-1), and max possible bits.",
    inputSchema: {
      type: "object",
      properties: {
        a: {
          type: "string",
          description: "First fingerprint (base64)",
        },
        b: {
          type: "string",
          description: "Second fingerprint (base64)",
        },
      },
      required: ["a", "b"],
    },
  },
  {
    name: "ladybug_xor_bind",
    description:
      "XOR bind two fingerprints (holographic composition). " +
      "Creates a new fingerprint dissimilar to both inputs.",
    inputSchema: {
      type: "object",
      properties: {
        a: {
          type: "string",
          description: "First fingerprint",
        },
        b: {
          type: "string",
          description: "Second fingerprint",
        },
      },
      required: ["a", "b"],
    },
  },
  {
    name: "ladybug_stats",
    description: "Get BindSpace statistics including node counts by zone.",
    inputSchema: {
      type: "object",
      properties: {},
    },
  },
  {
    name: "ladybug_sql",
    description: "Execute SQL query against DataFusion with cognitive UDFs.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "SQL query string",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "ladybug_cypher",
    description: "Execute Cypher graph query (transpiled to recursive CTEs).",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Cypher query string",
        },
      },
      required: ["query"],
    },
  },
];

// =============================================================================
// HTTP Client (JSON-RPC transport)
// =============================================================================

async function httpRequest(
  url: string,
  path: string,
  data: Record<string, unknown>
): Promise<unknown> {
  const response = await fetch(`${url}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`HTTP ${response.status}: ${error}`);
  }

  return response.json();
}

// =============================================================================
// Tool Handlers
// =============================================================================

async function handleTool(
  config: ServerConfig,
  name: string,
  args: Record<string, unknown>
): Promise<string> {
  const url = config.httpUrl;

  switch (name) {
    case "ladybug_encode": {
      const result = await httpRequest(url, "/api/v1/fingerprint", {
        content: args.text,
        data: args.data,
        style: args.style,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_bind": {
      const result = await httpRequest(url, "/api/v1/index", {
        address: args.address,
        fingerprint: args.fingerprint,
        label: args.label,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_read": {
      const result = await httpRequest(url, "/api/v1/read", {
        address: args.address,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_resonate": {
      const result = await httpRequest(url, "/api/v1/search/topk", {
        query: args.query,
        k: args.k || 10,
        threshold: args.threshold,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_hamming": {
      const result = await httpRequest(url, "/api/v1/hamming", {
        a: args.a,
        b: args.b,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_xor_bind": {
      const result = await httpRequest(url, "/api/v1/bind", {
        a: args.a,
        b: args.b,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_stats": {
      const result = await httpRequest(url, "/api/v1/stats", {});
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_sql": {
      const result = await httpRequest(url, "/api/v1/sql", {
        query: args.query,
      });
      return JSON.stringify(result, null, 2);
    }

    case "ladybug_cypher": {
      const result = await httpRequest(url, "/api/v1/cypher", {
        query: args.query,
      });
      return JSON.stringify(result, null, 2);
    }

    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

// =============================================================================
// MCP Server
// =============================================================================

async function main() {
  const config = DEFAULT_CONFIG;

  const server = new Server(
    {
      name: "ladybug-mcp",
      version: "0.4.0",
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // List tools
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return { tools: TOOLS };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      const result = await handleTool(config, name, args as Record<string, unknown>);
      return {
        content: [
          {
            type: "text",
            text: result,
          } as TextContent,
        ],
      };
    } catch (error) {
      return {
        content: [
          {
            type: "text",
            text: `Error: ${error instanceof Error ? error.message : String(error)}`,
          } as TextContent,
        ],
        isError: true,
      };
    }
  });

  // Start server
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error("LadybugDB MCP server running");
  console.error(`HTTP endpoint: ${config.httpUrl}`);
  console.error(`Flight endpoint: ${config.flightUrl || "disabled"}`);
}

main().catch(console.error);
