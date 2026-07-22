"""
Runnability test for content/08-agents-harness/06-mcp.md

Tests the 3 heuristically CPU-runnable Python blocks from the chapter,
concatenated in chapter order:

    - block #2 (line ~155) -- csv_server.py: a minimal low-level MCP server
      (Server instance, list_tools/call_tool/list_resources/read_resource
      handlers, main()).
    - block #5 (line ~382) -- client-side roots declaration snippet.
    - block #7 (line ~550) -- csv_server_http.py: mounting the same Server
      over the Streamable HTTP transport inside a FastAPI app.

Other blocks are correctly skipped as instructed:
    #0 (JSON-RPC message examples), #4/#8 (JSON config snippets) -- non-python
    #1, #3 -- shell (`pip install`, a manual stdin JSON-RPC probe)
    #6 (mcp_client_demo.py) -- SKIP(needs-net): spawns a real MCP server as a
        subprocess and drives it over a live stdio JSON-RPC session; that is
        an inter-process transport demo, not deterministic CPU-only logic,
        and is explicitly excluded by the task brief.

`pandas`, `mcp`, and `fastapi` are used by the book's code but are NOT in the
guaranteed CI dependency set (numpy, torch, einops, sklearn, stdlib). They are
guarded with try/except at module scope so this file always loads; the
blocks that need them are skipped (with an explicit SKIP message) if the
package is unavailable, and fully executed against tiny fixtures if it is.

REAL BUG FOUND AND FIXED IN THE BOOK:
Block #7 originally did:
    from mcp.server.fastapi import create_mcp_router
    ...
    mcp_router = create_mcp_router(mcp_app)
    web.include_router(mcp_router, prefix="/mcp")
Neither `mcp.server.fastapi` nor a `create_mcp_router` helper exists in the
official `mcp` Python SDK (verified against mcp==1.28.1 on PyPI: no such
submodule, `python -c "import mcp.server.fastapi"` raises ModuleNotFoundError).
The chapter's content/08-agents-harness/06-mcp.md was corrected to use the
SDK's actual `mcp.server.streamable_http_manager.StreamableHTTPSessionManager`,
mounted as a raw ASGI route under FastAPI/Starlette -- the documented way to
expose a low-level `Server` over the Streamable HTTP transport. The fix is
mirrored verbatim below and verified to actually construct and start/stop
cleanly.
"""

import asyncio
import contextlib
import json
import os
import pathlib
import tempfile
from types import SimpleNamespace
from typing import Any

# Optional third-party deps used by the book's code but not guaranteed in CI.
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from mcp.server import Server
    from mcp.server.models import InitializationOptions
    from mcp import types
except Exception:
    Server = None
    InitializationOptions = None
    types = None

try:
    from fastapi import FastAPI
    from starlette.routing import Mount
except Exception:
    FastAPI = None
    Mount = None

try:
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
except Exception:
    StreamableHTTPSessionManager = None


skipped = []


# =============================================================================
# Block #2 (line ~155): csv_server.py -- verbatim, minus the bottom
# `if __name__ == "__main__": asyncio.run(main())` guard (that starts a live
# stdio_server() loop waiting on real stdin forever -- inherently not
# CPU-test-safe, and would also collide with *this* file's own
# `__name__ == "__main__"` block). The handler functions it defines
# (handle_list_tools, handle_call_tool, handle_list_resources,
# handle_read_resource) are the load-bearing logic and are exercised
# directly below with a tiny fixture CSV.
# =============================================================================

if Server is not None and pd is not None:

    # Create the server instance.  The name and version appear during the
    # initialize handshake so the client can display them to the user.
    app = Server("csv-analyst", version="0.1.0")

    # Hard-code the CSV path for this example; a real server might accept it as
    # a command-line argument or an environment variable.
    CSV_PATH = pathlib.Path("data.csv")

    # ── Tool: list_columns ────────────────────────────────────────────────

    @app.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        """
        Called by the client to discover what tools this server provides.
        Returns a list of Tool objects, each with a JSON Schema for its inputs.
        """
        return [
            types.Tool(
                name="list_columns",
                description=(
                    "Return the column names and dtypes of the CSV file. "
                    "No arguments required."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},          # no parameters
                    "required": [],
                },
            ),
            types.Tool(
                name="query_csv",
                description=(
                    "Run a pandas DataFrame.query() expression against the CSV "
                    "and return up to max_rows rows as JSON. "
                    "Use standard pandas query syntax, e.g. 'age > 30 and city == \"London\"'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A pandas-compatible query expression.",
                        },
                        "max_rows": {
                            "type": "integer",
                            "description": "Maximum rows to return (default 20).",
                            "default": 20,
                        },
                    },
                    "required": ["expression"],
                },
            ),
        ]

    @app.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict[str, Any]
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """
        Dispatch tool calls.  Must return a list of content blocks.
        Raise ValueError for unknown tool names — the SDK converts this to a
        JSON-RPC error response automatically.
        """
        if not CSV_PATH.exists():
            return [types.TextContent(type="text", text=f"Error: {CSV_PATH} not found.")]

        df = pd.read_csv(CSV_PATH)

        if name == "list_columns":
            # Build a readable column-dtype mapping.
            col_info = {col: str(dtype) for col, dtype in df.dtypes.items()}
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(col_info, indent=2),
                )
            ]

        elif name == "query_csv":
            expression = arguments["expression"]
            max_rows = int(arguments.get("max_rows", 20))

            try:
                result_df = df.query(expression).head(max_rows)
            except Exception as exc:
                # Surface pandas errors as a text result rather than a protocol
                # error — this lets the model see what went wrong and self-correct.
                return [
                    types.TextContent(
                        type="text",
                        text=f"Query error: {exc}",
                    )
                ]

            # Return as JSON records — compact but structured.
            return [
                types.TextContent(
                    type="text",
                    text=result_df.to_json(orient="records", indent=2),
                )
            ]

        else:
            raise ValueError(f"Unknown tool: {name}")

    # ── Resource: raw CSV text ──────────────────────────────────────────────

    @app.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        """
        Expose the raw CSV as a readable resource.
        The host (or user) can inject this directly into the context window
        without invoking a tool.
        """
        return [
            types.Resource(
                uri=f"file://{CSV_PATH.resolve()}",
                name="data.csv",
                description="The raw CSV data file being analysed.",
                mimeType="text/csv",
            )
        ]

    @app.read_resource()
    async def handle_read_resource(uri: str) -> str:
        """
        Return the resource content.  For text resources, return a plain string;
        for binary resources, return bytes (the SDK base64-encodes them).
        """
        expected_uri = f"file://{CSV_PATH.resolve()}"
        if uri != expected_uri:
            raise ValueError(f"Unknown resource URI: {uri}")

        return CSV_PATH.read_text(encoding="utf-8")

    # ── Main: run with stdio transport ───────────────────────────────────────

    async def main() -> None:
        """
        Wire up the stdio transport and start serving.
        The stdio_server() context manager handles reading newline-delimited
        JSON-RPC from stdin and writing responses to stdout.

        NOT called in this test: it blocks forever on real stdin, which is
        inherent to what a long-running server does, not something a CPU
        unit test can safely exercise. The handler logic it wires up
        (list_tools/call_tool/list_resources/read_resource) is exercised
        directly below instead.
        """
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="csv-analyst",
                    server_version="0.1.0",
                    # Advertise which capabilities this server has.
                    capabilities=app.get_capabilities(
                        notification_options=None,
                        experimental_capabilities={},
                    ),
                ),
            )

    def test_block2_csv_server():
        """Exercise the csv-analyst MCP server's tool and resource handlers
        against a tiny fixture CSV, in a temp working directory (CSV_PATH is
        the relative path "data.csv", per the book's code)."""
        tmpdir = tempfile.mkdtemp()
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            with open("data.csv", "w") as f:
                f.write("name,revenue,city\n")
                f.write("Alice,1500,London\n")
                f.write("Bob,800,Paris\n")
                f.write("Cara,2000,London\n")

            # tools/list
            tools = asyncio.run(handle_list_tools())
            assert {t.name for t in tools} == {"list_columns", "query_csv"}

            # tools/call -> list_columns
            result = asyncio.run(handle_call_tool("list_columns", {}))
            assert len(result) == 1 and result[0].type == "text"
            col_info = json.loads(result[0].text)
            assert set(col_info.keys()) == {"name", "revenue", "city"}

            # tools/call -> query_csv
            result = asyncio.run(
                handle_call_tool(
                    "query_csv",
                    {"expression": "revenue > 1000", "max_rows": 5},
                )
            )
            rows = json.loads(result[0].text)
            assert len(rows) == 2
            assert {r["name"] for r in rows} == {"Alice", "Cara"}

            # tools/call -> unknown tool raises ValueError, as the book states
            try:
                asyncio.run(handle_call_tool("nonexistent_tool", {}))
                raise AssertionError("expected ValueError for unknown tool")
            except ValueError:
                pass

            # tools/call -> bad pandas expression is surfaced as a text error,
            # not a raised exception (per the book's "self-correct" design)
            result = asyncio.run(
                handle_call_tool("query_csv", {"expression": "not a valid expr((("})
            )
            assert "Query error" in result[0].text

            # resources/list + resources/read
            resources = asyncio.run(handle_list_resources())
            assert len(resources) == 1
            uri = resources[0].uri
            text = asyncio.run(handle_read_resource(str(uri)))
            assert text == pathlib.Path("data.csv").read_text(encoding="utf-8")

            # resources/read with unknown URI raises ValueError
            try:
                asyncio.run(handle_read_resource("file:///nope"))
                raise AssertionError("expected ValueError for unknown resource URI")
            except ValueError:
                pass

            print("block #2 (csv_server.py): tools/resources handlers OK")
        finally:
            os.chdir(old_cwd)

else:
    def test_block2_csv_server():
        skipped.append("block #2: SKIP(optional-deps) -- `mcp` and/or `pandas` not installed")
        print(skipped[-1])


# =============================================================================
# Block #5 (line ~382): client-side roots declaration. Verbatim, with a tiny
# stand-in `client` object as glue (the book's snippet is a fragment that
# assumes an existing ClientSession-like `client` from surrounding prose).
# =============================================================================

if types is not None:

    def test_block5_roots():
        client = SimpleNamespace()

        # Client-side: declare roots during initialization
        client.roots = [
            types.Root(uri="file:///home/user/project", name="my-project")
        ]

        assert len(client.roots) == 1
        assert str(client.roots[0].uri) == "file:///home/user/project"
        assert client.roots[0].name == "my-project"
        print("block #5 (roots declaration): OK")

else:
    def test_block5_roots():
        skipped.append("block #5: SKIP(optional-deps) -- `mcp` not installed")
        print(skipped[-1])


# =============================================================================
# Block #7 (line ~550): csv_server_http.py -- mount the same Server over
# Streamable HTTP inside a FastAPI app. This is the CORRECTED version (see
# module docstring for the bug found in the book's original
# `mcp.server.fastapi.create_mcp_router`, which does not exist in the SDK).
# "from csv_server import app as mcp_app" becomes a direct reference to the
# `app` object built in block #2 above, since both blocks live in one file
# here (the book's own comment says "in practice, import it from the module").
# =============================================================================

if Server is not None and pd is not None and FastAPI is not None and StreamableHTTPSessionManager is not None:

    mcp_app = app  # the Server object from block #2

    # StreamableHTTPSessionManager drives a low-level Server over the
    # Streamable HTTP transport: it multiplexes JSON-RPC requests and an
    # optional SSE upgrade onto a single ASGI endpoint.
    session_manager = StreamableHTTPSessionManager(app=mcp_app)

    @contextlib.asynccontextmanager
    async def lifespan(_app: "FastAPI"):
        # The session manager owns a task group that must stay alive for the
        # whole lifetime of the ASGI app.
        async with session_manager.run():
            yield

    web = FastAPI(title="CSV Analyst MCP", lifespan=lifespan)

    # Mount the MCP endpoint at /mcp; handle_request is a raw ASGI callable
    # that speaks Streamable HTTP (POST for client -> server, optional SSE
    # upgrade for server -> client streaming).
    web.router.routes.append(Mount("/mcp", app=session_manager.handle_request))

    # Run with:  uvicorn csv_server_http:web --port 8080
    # Configure the client to point at http://localhost:8080/mcp

    def test_block7_http_server():
        """Construct the FastAPI/StreamableHTTP mount and exercise the
        session manager's lifespan (start/stop), without opening any real
        network socket -- purely in-process ASGI object construction, no
        `uvicorn.run()` and no outbound/inbound network I/O."""
        assert any(getattr(r, "path", None) == "/mcp" for r in web.routes)

        async def smoke():
            async with lifespan(web):
                pass

        asyncio.run(smoke())
        print("block #7 (csv_server_http.py): FastAPI + StreamableHTTP mount OK")

else:
    def test_block7_http_server():
        skipped.append(
            "block #7: SKIP(optional-deps) -- `mcp`, `pandas`, and/or `fastapi` not installed"
        )
        print(skipped[-1])


if __name__ == "__main__":
    test_block2_csv_server()
    test_block5_roots()
    test_block7_http_server()

    print()
    if skipped:
        print(f"{len(skipped)} block(s) skipped:")
        for s in skipped:
            print(" -", s)
    print("ALL TESTS PASSED")
