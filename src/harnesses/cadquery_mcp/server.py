"""
MCP server: a thin wrapper that exposes one persistent CadSession as tools.

The whole point of "strong agentic" lives here: SESSION is a module-level
singleton, and the stdio server is one long-lived process, so the CadQuery
namespace and undo history survive across tool calls. Each tool just forwards
to a CadSession method.

Dict-returning tools become a JSON text block the model reads. render_views
returns a list of Image objects (one per view) so the model actually SEES the
renders -- it must NOT be annotated `-> dict`, or FastMCP would build an output
schema and reject the list (see func_metadata._convert_to_content).
"""

import os
import sys
import tempfile
import contextlib

# Repo root on the import path so `src.*` imports resolve no matter how the
# server is launched (three levels up from src/harnesses/cadquery_mcp).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mcp.server.fastmcp import FastMCP, Image

from src.harnesses.cadquery_mcp.cad_session import CadSession

mcp = FastMCP("neuralcad", json_response=True)

# One session for the whole server process -> state persists between tool calls.
SESSION = CadSession()


@contextlib.contextmanager
def _quiet():
    """Keep library print()s off stdout.

    On a stdio MCP server stdout IS the JSON-RPC channel, but our reused
    helpers (export_as_image / export_as_step) print progress there. Redirect
    that to stderr so it shows up as a server log instead of corrupting the
    protocol. (run_step captures its own stdout already.)
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


@mcp.tool()
def run_step(code: str) -> dict:
    """Run one incremental CadQuery snippet against the persistent model.

    Variables persist between calls; build the edit step by step and keep the
    current model in the variable `result`. On error the step is rolled back.
    Returns ok/stdout/info (bbox, size, face & edge counts) or the error.
    """
    return SESSION.run_step(code)


@mcp.tool()
def inspect() -> dict:
    """Inspect the current model: bounding box, size, center, face/edge counts.

    Use this instead of guessing dimensions or face indices.
    """
    return SESSION.inspect()


@mcp.tool()
def undo() -> dict:
    """Revert the most recent successful step (Ctrl-Z)."""
    return SESSION.undo()


@mcp.tool()
def reset() -> dict:
    """Wipe all state and history and start over from an empty session."""
    return SESSION.reset()


@mcp.tool()
def export_step(output_dir: str) -> dict:
    """Export the current model to <output_dir>/tmp.step."""
    with _quiet():
        return SESSION.export_step(output_dir)


@mcp.tool()
def render_views(views: list[str] | None = None, highlight_faces: list[list[float]] | None = None, highlight_edges: list[list[float]] | None = None):
    """Render the current model and return the image(s) so you can see it.

    `views` is a list of named viewpoints to choose from: toprightiso, front,
    back, left, right, top, bottom. None -> just the iso view. Returns one
    image per requested view. To save on context, you should only render the necessary views.

    To verify a selection before you operate on it, pass the `center` of a face
    from list_faces as highlight_faces=[[x, y, z]] (or an edge center from
    list_edges as highlight_edges=[[x, y, z]]). Each is drawn in red on the
    render -- the same NearestToPointSelector you would use in run_step -- so you
    can confirm you picked the right entity before fillet/chamfer. Both accept a
    list, so you can mark several at once.
    """
    with _quiet():
        result = SESSION.render(tempfile.mkdtemp(prefix="cad_render_"), views=views, highlight_faces=highlight_faces, highlight_edges=highlight_edges)
    if not result.get("ok"):
        return result  # error dict -> text block with the reason / valid views
    return [Image(path=p) for p in result["png_paths"]] + [
        {"views": views or ["toprightiso"]}
    ]

@mcp.tool()
def list_faces(limit: int = 50) -> dict:
    """List the faces of the current model: index, type (PLANE/CYLINDER/...),
    area, center and normal. Sorted by area, largest first; `limit` caps the
    entries (n_faces is always the true total).

    To operate on a listed face, select it by its center in run_step:
    result.faces(cq.selectors.NearestToPointSelector((x, y, z))). Do NOT
    select by index - indices change whenever the model is rebuilt.
    """
    return SESSION.list_faces(limit=limit)


@mcp.tool()
def list_edges(limit: int = 50) -> dict:
    """List the edges of the current model: index, type (LINE/CIRCLE/...),
    length, center, and radius for circular edges. Sorted by length, longest
    first; `limit` caps the entries (n_edges is always the true total).

    To operate on a listed edge (e.g. fillet/chamfer), select it by its center
    in run_step: result.edges(cq.selectors.NearestToPointSelector((x, y, z))).
    Do NOT select by index - indices change whenever the model is rebuilt.
    """
    return SESSION.list_edges(limit=limit)


# Run with stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")
