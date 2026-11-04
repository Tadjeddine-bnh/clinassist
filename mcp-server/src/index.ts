import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
 
  
import { z } from "zod";

// Where the Python API runs. Overridable later, e.g. by Claude Desktop's config.
const API_URL = process.env.CLINASSIST_API_URL ?? "http://127.0.0.1:8000";

type ToolReply = { content: { type: "text"; text: string }[]; isError?: boolean };

// The only place that talks to the Python API.
async function callTool(name: string, args: Record<string, unknown>): Promise<ToolReply> {
  try {
    const res = await fetch(`${API_URL}/tools/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ arguments: args }),
      signal: AbortSignal.timeout(60_000),   // the X-ray model can take a while on first use
    });
    if (!res.ok) {
      const detail = await res.text();
      return { isError: true, content: [{ type: "text", text: `ClinAssist API error ${res.status}: ${detail}` }] };
    }
    const data = (await res.json()) as { result: string };
    return { content: [{ type: "text", text: data.result }] };
  } catch (err) {
    return {
      isError: true,
      content: [{ type: "text", text: `Could not reach the ClinAssist API at ${API_URL}. Is it running? (${String(err)})` }],
    };
  }
}

const server = new McpServer({ name: "clinassist", version: "0.1.0" });
const READ_ONLY = { readOnlyHint: true, openWorldHint: false };

server.registerTool(
  "search_guidelines",
  {
    title: "Search clinical guidelines",
    description:
      "Search the WHO Pocket Book of Hospital Care for Children and any indexed patient documents. " +
      "Returns passages, each starting with a label like [source: file.pdf, p. 159].",
    inputSchema: {
      query: z.string().min(1).describe("A short clinical phrase, e.g. 'IV fluids severe dehydration child'"),
    },
    annotations: READ_ONLY,
  },
  async ({ query }) => callTool("search_guidelines", { query }),
);

server.registerTool(
  "list_files",
  {
    title: "List documents",
    description:
      "List the patient documents, reports and dictation transcripts that can be opened with read_file. " +
      "Call this first when a question is about a specific document.",
    annotations: READ_ONLY,
  },
  async () => callTool("list_files", {}),
);

server.registerTool(
  "read_file",
  {
    title: "Read a document",
    description: "Read one document in full. file_name must be exactly a name returned by list_files.",
    inputSchema: { file_name: z.string().min(1).describe("Exact file name from list_files") },
    annotations: READ_ONLY,
  },
  async ({ file_name }) => callTool("read_file", { file_name }),
);

server.registerTool(
  "list_images",
  {
    title: "List images",
    description: "List the image files that can be analysed with classify_xray. Call this first for X-ray questions.",
    annotations: READ_ONLY,
  },
  async () => callTool("list_images", {}),
);

server.registerTool(
  "classify_xray",
  {
    title: "Classify a chest X-ray",
    description:
      "Run the chest X-ray classifier on one image. It only separates NORMAL from PNEUMONIA. " +
      "It cannot detect tuberculosis or any other condition, and it is not validated for clinical use.",
    inputSchema: { file_name: z.string().min(1).describe("Exact image name from list_images") },
    annotations: READ_ONLY,
  },
  async ({ file_name }) => callTool("classify_xray", { file_name }),
);

await server.connect(new StdioServerTransport());
console.error(`clinassist MCP server running, using API at ${API_URL}`);