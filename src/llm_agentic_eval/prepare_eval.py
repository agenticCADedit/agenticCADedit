# this file has been created using GPT5.6 sol
import argparse
import json
import os
import random
import shutil
import string
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cadquery as cq
import numpy as np
import vtk
from PIL import Image, ImageDraw, ImageFont
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db import DatabaseManager
from src.utils.process_config import load_config


CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "edit_192_external.json"
SEED = "neuralcad-agentic-eval"
VIEWS = ["toprightiso", "front", "back", "left", "right", "top", "bottom"]
TILE_SIZE = 512
GRID_COLUMNS = 4
GRID_ROWS = 2
LABEL_FONT = ImageFont.load_default(size=20)


RATER_PROMPT = """# Blind rating of CAD edits

Rate every unfinished task in this directory. Treat every task as an
independent evaluation.

Use one fresh subagent or isolated worker per open task. Give that worker only
the path to `tasks/<id>/task.md`. The worker must inspect the referenced images
and write only `ratings/<id>.json`. The worker should work efficiently, usage is limited. 
Several independent tasks may be evaluated
in parallel. The parent agent dispatches tasks, checks that each result is valid
JSON in the requested format, and retries missing or malformed results; it does
not rescore the candidates itself.

If the environment has no subagent support, process one task at a time and
treat each task as a fresh, isolated evaluation.

For every directory in `tasks/` that has no matching `ratings/<id>.json`:

1. Read `tasks/<id>/task.md`.
2. Inspect `start.png`, then `reference.png`, then every candidate image.
3. Score every candidate on both 1-7 scales from `task.md`.
4. Write the result to `ratings/<id>.json` in the exact format shown there.

`reference.png` is the requesting human's solution. It is useful for
understanding the intended geometry, but it is not rated and is not the only
possible correct solution. Candidates may legitimately be identical to each
other or to the reference; score each candidate from its visible result and do
not penalize duplicates. Leave existing rating files unchanged so the work can
be resumed later.
"""


def brep_step_path(db, brep_id):
    brep = db.breps.find_one({"_id": brep_id}) or {}
    paths = brep.get("step") or brep.get("stp") or []
    return Path(db.root_dir) / paths[0] if paths else None


def create_contact_sheet(db, brep_id, destination):
    step_path = brep_step_path(db, brep_id)
    if step_path is None:
        return False

    shape = cq.importers.importStep(str(step_path)).val()
    vertices, triangle_indices = shape.tessellate(1.0, 0.5)
    points = np.asarray([vertex.toTuple() for vertex in vertices], dtype=float)
    triangle_ids = np.asarray(triangle_indices, dtype=np.int64)

    if points.size == 0 or triangle_ids.size == 0:
        return False

    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(np.linalg.norm(points - center, axis=1).max(), 1e-6)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(points, deep=True))
    triangle_cells = vtk.vtkCellArray()
    triangle_cells.SetData(
        numpy_to_vtk(
            np.arange(0, 3 * len(triangle_ids) + 1, 3, dtype=np.int64),
            deep=True,
        ),
        numpy_to_vtk(triangle_ids.ravel(), deep=True),
    )
    surface = vtk.vtkPolyData()
    surface.SetPoints(vtk_points)
    surface.SetPolys(triangle_cells)
    surface_mapper = vtk.vtkPolyDataMapper()
    surface_mapper.SetInputData(surface)
    surface_actor = vtk.vtkActor()
    surface_actor.SetMapper(surface_mapper)
    surface_actor.GetProperty().SetColor(0.78, 0.78, 0.78)
    surface_actor.GetProperty().SetAmbient(0.55)
    surface_actor.GetProperty().SetDiffuse(0.45)
    surface_actor.GetProperty().SetSpecular(0)
    surface_actor.GetProperty().SetInterpolationToFlat()

    edge_points = vtk.vtkPoints()
    edge_cells = vtk.vtkCellArray()
    point_index = 0
    for edge in shape.Edges():
        polyline, _ = edge.sample(0.5)
        if len(polyline) < 2:
            continue
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(polyline))
        for index, point in enumerate(polyline):
            edge_points.InsertNextPoint(*point.toTuple())
            line.GetPointIds().SetId(index, point_index)
            point_index += 1
        edge_cells.InsertNextCell(line)

    edges = vtk.vtkPolyData()
    edges.SetPoints(edge_points)
    edges.SetLines(edge_cells)
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputData(edges)
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetColor(0.28, 0.28, 0.28)
    edge_actor.GetProperty().SetLineWidth(1.2)
    edge_actor.GetProperty().LightingOff()

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1, 1, 1)
    renderer.AddActor(surface_actor)
    renderer.AddActor(edge_actor)

    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(True)
    render_window.SetSize(TILE_SIZE * 2, TILE_SIZE * 2)
    render_window.SetMultiSamples(8)
    render_window.AddRenderer(renderer)

    camera = renderer.GetActiveCamera()
    camera.SetFocalPoint(*center)
    camera.ParallelProjectionOn()
    camera.SetParallelScale(radius / 0.84)

    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(render_window)
    capture.SetInputBufferTypeToRGB()

    view_directions = [
        ((1, -1, 1), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 0)),
        ((0, 0, -1), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0)),
        ((1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 1)),
        ((0, -1, 0), (0, 0, 1)),
    ]

    sheet = Image.new(
        "RGB",
        (GRID_COLUMNS * TILE_SIZE, GRID_ROWS * TILE_SIZE),
        "white",
    )
    for index, (view_name, (direction, up)) in enumerate(
        zip(VIEWS, view_directions)
    ):
        direction = np.asarray(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        camera.SetPosition(*(center + direction * radius * 4))
        camera.SetViewUp(*up)
        camera.OrthogonalizeViewUp()
        renderer.ResetCameraClippingRange()
        render_window.Render()
        capture.Modified()
        capture.Update()

        image = capture.GetOutput()
        width, height, _ = image.GetDimensions()
        pixels = vtk_to_numpy(image.GetPointData().GetScalars())
        pixels = np.flipud(pixels.reshape(height, width, -1)[..., :3])
        tile = Image.fromarray(pixels).resize(
            (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS
        )
        label = ImageDraw.Draw(tile)
        text_box = label.textbbox((12, 10), view_name, font=LABEL_FONT)
        label.rectangle(
            (
                text_box[0] - 5,
                text_box[1] - 3,
                text_box[2] + 5,
                text_box[3] + 3,
            ),
            fill="white",
        )
        label.text((12, 10), view_name, fill=(40, 40, 40), font=LABEL_FONT)
        sheet.paste(
            tile,
            (
                (index % GRID_COLUMNS) * TILE_SIZE,
                (index // GRID_COLUMNS) * TILE_SIZE,
            ),
        )

    render_window.Finalize()

    grid = ImageDraw.Draw(sheet)
    for column in range(GRID_COLUMNS + 1):
        x = min(column * TILE_SIZE, sheet.width - 1)
        grid.line((x, 0, x, sheet.height - 1), fill=(160, 160, 160), width=1)
    for row in range(GRID_ROWS + 1):
        y = min(row * TILE_SIZE, sheet.height - 1)
        grid.line((0, y, sheet.width - 1, y), fill=(160, 160, 160), width=1)

    sheet.save(destination)
    return True


def request_instruction(request):
    segments = request.get("corrected_transcript_segments") or []
    if not segments:
        segments = (request.get("transcript") or {}).get("segments", [])
    if segments:
        return " ".join(segment["text"].strip() for segment in segments)
    return (request.get("text") or "").strip()


def task_markdown(task_id, request, candidate_labels):
    instruction = request_instruction(request)
    if not instruction:
        raise ValueError(f"No instruction found for {task_id}")

    candidates = "\n".join(
        f"  - `{label}.png`" for label in candidate_labels
    )
    rating_template = {
        "task": task_id,
        "ratings": {
            label: {"instruction": 1, "quality": 1, "note": ""}
            for label in candidate_labels
        },
    }

    return f"""# Task {task_id}

* difficulty: {request.get("difficulty", "")}
* modality: {request.get("modality", "")}

## Instruction

> {instruction}

## Images

Every image shows one model from seven views
(`toprightiso`, `front`, `back`, `left`, `right`, `top`, `bottom`).

- `start.png` - the model **before** the edit
- `reference.png` - the requesting human's solution (reference, **not rated**)
- the candidates to rate:
{candidates}

## What to write

`../../ratings/{task_id}.json`:

```json
{json.dumps(rating_template, ensure_ascii=False)}
```

## Scales (both 1-7)

**instruction** - how well was the requested edit understood and performed?

1. Wrong direction or makes the model worse.
2. No meaningful edit.
3. Rough attempt with major errors or omissions.
4. Mostly correct with noticeable errors or omissions.
5. Correct with small errors or omissions.
6. Precise and complete.
7. Perfect, including useful thoughtful extras.

**quality** - how good is the resulting CAD model?

1. No usable model.
2. Invalid or severely broken geometry.
3. Poor or overly simplistic result.
4. Acceptable first pass.
5. Good result with room for improvement.
6. Professional quality.
7. Exceptionally polished result.
"""


def candidate_record(db, arm, edit):
    if edit is None:
        return {
            "arm": arm,
            "edit_user": arm,
            "edit_id": None,
            "brep_end": None,
            "valid": False,
            "failed_run": True,
        }

    failed = bool(edit.get("failed_run"))
    return {
        "arm": arm,
        "edit_user": edit["user"],
        "edit_id": edit["_id"],
        "brep_end": edit.get("brep_end"),
        "valid": (
            not failed
            and brep_step_path(db, edit.get("brep_end")) is not None
        ),
        "failed_run": failed,
    }


def prepare_task(db, task_id, request, edits, users, model_arms, tasks_dir):
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True)

    reference = next(
        edit for edit in edits if edit["user"] == request["user"]
    )
    other_human = next(
        (
            edit
            for edit in edits
            if edit["user"] != request["user"]
            and users.get(edit["user"], {}).get("is_human", True)
        ),
        None,
    )

    if not create_contact_sheet(
        db, request["brep_start"], task_dir / "start.png"
    ):
        raise RuntimeError(f"Could not render start model for {task_id}")
    if not create_contact_sheet(
        db, reference["brep_end"], task_dir / "reference.png"
    ):
        raise RuntimeError(f"Could not render reference model for {task_id}")

    candidates = [candidate_record(db, "human-baseline", other_human)]
    for arm in model_arms:
        edit = next((edit for edit in edits if edit["user"] == arm), None)
        candidates.append(candidate_record(db, arm, edit))
    random.Random(f"{SEED}:{request['_id']}").shuffle(candidates)

    output_task = {
        "request_id": request["_id"],
        "reference_edit_id": reference["_id"],
    }
    visible_labels = []
    for label, candidate in zip(string.ascii_uppercase, candidates):
        if candidate["valid"]:
            candidate["valid"] = create_contact_sheet(
                db, candidate["brep_end"], task_dir / f"{label}.png"
            )
        output_task[label] = candidate

        if candidate["valid"]:
            visible_labels.append(label)

    (task_dir / "task.md").write_text(
        task_markdown(task_id, request, visible_labels),
        encoding="utf-8",
    )
    return output_task


def main():
    if os.environ.get("LLM_AGENTIC_EVAL_XVFB") != "1":
        environment = os.environ.copy()
        environment["LLM_AGENTIC_EVAL_XVFB"] = "1"
        raise SystemExit(
            subprocess.call(
                ["xvfb-run", "-a", sys.executable, *sys.argv], env=environment
            )
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-tasks", type=int)
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    db = DatabaseManager(config)

    requests = sorted(
        db.requests.find({"request_type": "edit"}),
        key=lambda request: request["_id"],
    )
    random.Random(SEED).shuffle(requests)
    if args.num_tasks is not None:
        requests = requests[:args.num_tasks]

    users = {user["_id"]: user for user in db.users.find({})}
    edits_by_request = defaultdict(list)
    available_edit_users = set()
    for edit in db.edits.find({}):
        edits_by_request[edit.get("request")].append(edit)
        available_edit_users.add(edit.get("user"))

    model_arms = [
        arm
        for arm in config["benchmark_eval_users"]["edit"]
        if arm not in {"gt human", "other human"}
        and arm in available_edit_users
    ]

    tasks_dir = args.output_dir / "tasks"
    ratings_dir = args.output_dir / "ratings"

    shutil.rmtree(tasks_dir, ignore_errors=True)
    tasks_dir.mkdir(parents=True)
    shutil.rmtree(args.output_dir / "key", ignore_errors=True)
    ratings_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "RATER_PROMPT.md").write_text(
        RATER_PROMPT, encoding="utf-8"
    )

    tasks = {}
    for index, request in enumerate(requests, start=1):
        task_id = f"t{index - 1:03d}"
        tasks[task_id] = prepare_task(
            db,
            task_id,
            request,
            edits_by_request[request["_id"]],
            users,
            model_arms,
            tasks_dir,
        )
        print(f"Prepared {index}/{len(requests)}: {task_id}")

    key = {"arms": model_arms, "seed": SEED, "tasks": tasks}
    key_path = args.output_dir.parent / f"{args.output_dir.name}_key.json"
    key_path.write_text(
        json.dumps(key, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nPrepared {len(tasks)} tasks in {args.output_dir}")
    print(f"Private key: {key_path}")


if __name__ == "__main__":
    main()
