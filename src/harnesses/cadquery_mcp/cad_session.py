"""
CadSession - a stateful, evolving CadQuery environment ("strong agentic").

Unlike src/harnesses/cadquery_script.py (which builds a FRESH namespace on every
call, execs a whole my_cad_function, and throws the namespace away), this class
keeps ONE namespace alive across many run_step() calls. Variables defined in one
step are visible in the next, so the model builds the edit incrementally.

A history stack of namespace snapshots provides undo (Ctrl-Z). This module knows
nothing about MCP - it is plain Python and can be tested on its own.

Convention: the current model lives in the variable `result`. render(),
export_step() and inspect() all operate on namespace["result"].
"""

import os
import sys
import io
import contextlib
import traceback

import cadquery as cq
from cadquery import exporters

# Repo root on the import path so the project helpers below resolve no matter
# how this module is launched (three levels up from src/harnesses/cadquery_mcp).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))


class CadSession:
    def __init__(self):
        self.namespace = self._fresh_namespace()
        self.history = []  # stack of shallow namespace snapshots, for undo()

    def _fresh_namespace(self) -> dict:
        # Same seed as load_and_execute_function() in cadquery_script.py: the
        # exec'd code can use cq / Workplane / etc. without importing them.
        return {
            "cq": cq,
            "cadquery": cq,
            "Workplane": cq.Workplane,
            "Assembly": cq.Assembly,
            "exporters": exporters,
            "os": os,
            "sys": sys,
            "__builtins__": __builtins__,
        }

    # --- the core: run one incremental step ---------------------------------
    def run_step(self, code: str) -> dict:
        """Exec a snippet of CadQuery code against the persistent namespace.

        Snapshots the namespace first so a failed step rolls back cleanly and a
        successful step can be undone later. Captures stdout so the model can
        print() things to inspect the model.
        """
        # Snapshot BEFORE running. Shallow copy = current name->object bindings;
        # enough because CadQuery ops reassign (result = result.fillet(...))
        # rather than mutate in place.
        self.history.append(self.namespace.copy())

        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                exec(code, self.namespace)
        except Exception as e:
            # Step failed midway -> restore the pre-step state and drop the
            # snapshot (this step never happened).
            self.namespace = self.history.pop()
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "stdout": out.getvalue(),
            }

        # Success -> keep the snapshot in history so undo() can revert this step.
        return {
            "ok": True,
            "stdout": out.getvalue(),
            "info": self._describe(),
            "n_steps": len(self.history),
        }

    def undo(self) -> dict:
        """Revert the most recent successful step (Ctrl-Z)."""
        if not self.history:
            return {"ok": False, "error": "Nothing to undo (history is empty)."}
        self.namespace = self.history.pop()
        return {"ok": True, "info": self._describe(), "n_steps": len(self.history)}

    def reset(self) -> dict:
        """Wipe all state and history - start over."""
        self.namespace = self._fresh_namespace()
        self.history = []
        return {"ok": True}

    # --- querying the current model (so the model asks instead of guessing) --
    def inspect(self) -> dict:
        """Return bounding box, size and face/edge counts of the current model."""
        return self._describe()
    
    def list_faces(self, limit: int = 50) -> dict:
        """List every face of the current model: type, area, center, normal (sorted and truncated)."""
        shape = self._current_shape()
        if shape is None:
            return {"result": "no 'result' variable set yet"}
        try:
            face_information = []
            for i, face in enumerate(shape.Faces()):
                # normalAt can fail on exotic surfaces; one bad face must not
                # kill the whole listing.
                try:
                    nx, ny, nz = face.normalAt().toTuple()
                    normal = {"x": round(nx, 3), "y": round(ny, 3), "z": round(nz, 3)}
                except Exception:
                    normal = None
                cx, cy, cz = face.Center().toTuple()
                face_information.append({
                    "index": i,
                    "type": face.geomType(),
                    "area": round(face.Area(), 3),
                    "center": {"x": round(cx, 3), "y": round(cy, 3), "z": round(cz, 3)},
                    "normal": normal,
                })

            n_faces = len(face_information)
            face_information.sort(key=lambda f: f["area"], reverse=True)
            return {
                "result": "present",
                "n_faces": n_faces,
                "truncated": n_faces > limit,
                "faces": face_information[:limit],
            }
        except Exception as e:
            return {"result": "present but could not inspect", "error": f"{type(e).__name__}: {e}"}

    def list_edges(self, limit: int = 50) -> dict:
        """List every edge of the current model: type, length, center, radius."""
        shape = self._current_shape()
        if shape is None:
            return {"result": "no 'result' variable set yet"}
        try:
            edge_information = []
            for i, edge in enumerate(shape.Edges()):
                cx, cy, cz = edge.Center().toTuple()
                entry = {
                    "index": i,
                    "type": edge.geomType(),
                    "length": round(edge.Length(), 3),
                    "center": {"x": round(cx, 3), "y": round(cy, 3), "z": round(cz, 3)},
                }
                # radius() raises ValueError on non-circular edges
                try:
                    entry["radius"] = round(edge.radius(), 3)
                except Exception:
                    pass
                edge_information.append(entry)

            n_edges = len(edge_information)
            edge_information.sort(key=lambda e: e["length"], reverse=True)
            return {
                "result": "present",
                "n_edges": n_edges,
                "truncated": n_edges > limit,
                "edges": edge_information[:limit],
            }
        except Exception as e:
            return {"result": "present but could not inspect", "error": f"{type(e).__name__}: {e}"}

    def _current_shape(self):
        """Extract a cq.Shape from namespace['result'] (Workplane/Assembly/Shape)."""
        result = self.namespace.get("result")
        if result is None:
            return None
        if isinstance(result, cq.Assembly):
            return result.toCompound()
        if hasattr(result, "val"):  # Workplane
            return result.val()
        return result  # already a Shape

    def _describe(self) -> dict:
        shape = self._current_shape()
        if shape is None:
            return {"result": "no 'result' variable set yet"}
        try:
            bb = shape.BoundingBox()
            return {
                "result": "present",
                "n_faces": len(shape.Faces()),
                "n_edges": len(shape.Edges()),
                "bbox": {
                    "xmin": round(bb.xmin, 3), "xmax": round(bb.xmax, 3),
                    "ymin": round(bb.ymin, 3), "ymax": round(bb.ymax, 3),
                    "zmin": round(bb.zmin, 3), "zmax": round(bb.zmax, 3),
                },
                "size": {"x": round(bb.xlen, 3), "y": round(bb.ylen, 3), "z": round(bb.zlen, 3)},
                "center": {"x": round(bb.center.x, 3), "y": round(bb.center.y, 3), "z": round(bb.center.z, 3)},
            }
        except Exception as e:
            return {"result": "present but could not inspect", "error": f"{type(e).__name__}: {e}"}

    # --- outputs ------------------------------------------------------------
    # Named viewpoints the model may ask for (reused from cadquery_rendering).
    AVAILABLE_VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]

    def render(self, output_dir: str, views=None, highlight_faces=None, highlight_edges=None) -> dict:
        """Render the current model to PNG(s) from one or more named viewpoints.

        Lets the model look at the part from several sides instead of one fixed
        angle. `views` is a single view name or a list from AVAILABLE_VIEWS
        (toprightiso, front, back, left, right, top, bottom); None -> iso only.
        Reuses export_as_image, which auto-starts Xvfb on headless Linux and
        writes one tmp_<view>.png per view. Returns the list of PNG paths so the
        MCP layer can hand the images back to the model.
        """
        result = self.namespace.get("result")
        if result is None:
            return {"ok": False, "error": "No 'result' to render."}

        # Resolve each highlight coordinate to the nearest face/edge -- the same
        # NearestToPointSelector the model uses in run_step -- so the render marks
        # the exact entity it is about to operate on and the pick gets verified.
        shape = self._current_shape()
        highlight = []
        for c in highlight_faces or []:
            highlight += cq.selectors.NearestToPointSelector(tuple(c)).filter(shape.Faces())
        for c in highlight_edges or []:
            highlight += cq.selectors.NearestToPointSelector(tuple(c)).filter(shape.Edges())

        from src.utils.cadquery_rendering import export_as_image, VIEW_PROJECTIONS

        if isinstance(views, str):
            views = [views]
        unknown = [v for v in (views or []) if v not in VIEW_PROJECTIONS]
        if unknown:
            return {
                "ok": False,
                "error": f"Unknown view(s): {unknown}",
                "available_views": self.AVAILABLE_VIEWS,
            }

        os.makedirs(output_dir, exist_ok=True)
        try:
            paths = export_as_image(result, output_dir, views=views, highlight=highlight)
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "hint": "Offscreen rendering needs Xvfb on Linux (run under xvfb-run).",
            }
        if not paths:
            return {
                "ok": False,
                "error": "Rendering produced no images.",
                "hint": "Offscreen rendering needs Xvfb on Linux (apt-get install -y xvfb).",
            }
        return {"ok": True, "png_paths": paths, "available_views": self.AVAILABLE_VIEWS}

    def export_step(self, output_dir: str) -> dict:
        """Export the current model to <output_dir>/tmp.step (reuses cadquery_script)."""
        result = self.namespace.get("result")
        if result is None:
            return {"ok": False, "error": "No 'result' to export."}
        from src.harnesses.cadquery_script import export_as_step
        os.makedirs(output_dir, exist_ok=True)
        export_as_step(result, output_dir)
        return {"ok": True, "step_path": os.path.join(output_dir, "tmp.step")}
