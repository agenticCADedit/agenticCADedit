import argparse
import json
import string
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.db import DatabaseManager
from src.utils.process_config import load_config


CONFIG_PATH = PROJECT_ROOT / "src" / "config" / "edit_192_external.json"
RATER_USER = "claude-opus_agentic-eval-rating"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_dir", type=Path)
    args = parser.parse_args()

    key_path = args.eval_dir.parent / f"{args.eval_dir.name}_key.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))

    records = []
    for task_id, task in key["tasks"].items():
        result = json.loads(
            (args.eval_dir / "ratings" / f"{task_id}.json").read_text(
                encoding="utf-8"
            )
        )

        for label, candidate in task.items():
            if label not in string.ascii_uppercase:
                continue

            if candidate["valid"]:
                rating = result["ratings"][label]
                score_instr = rating["instruction"]
                score_quality = rating["quality"]
                comments = rating["note"]
            else:
                score_instr = 2
                score_quality = 1
                comments = "Invalid or failed run."

            records.append(
                {
                    "edit": candidate["edit_id"],
                    "score_instr": score_instr,
                    "score_quality": score_quality,
                    "comments": comments,
                }
            )

    db = DatabaseManager(load_config(CONFIG_PATH))
    db.insert_user(RATER_USER, is_human=False)

    for record in records:
        existing = db.ratings.find_one(
            {"user": RATER_USER, "edit": record["edit"]}
        )
        values = {key: value for key, value in record.items() if key != "edit"}
        if existing:
            db.ratings.update_one(
                {"_id": existing["_id"]}, {"$set": values}
            )
        else:
            db.ratings.insert_one(
                {"user": RATER_USER, **record}
            )

    print(f"Imported {len(records)} ratings as {RATER_USER}")


if __name__ == "__main__":
    main()