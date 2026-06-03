from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data.json"
DEFAULT_CHECKPOINT = Path("/home/ubuntu/TASTE_Checkpoint")


def load_scorer(checkpoint: Path):
    sys.path.insert(0, "/home/ubuntu/taste/taste-scorer/src")
    from taste_scorer import PreferenceScorer

    return PreferenceScorer.from_checkpoint(checkpoint, device="cuda")


def build_ordered_pair_frame(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for prompt in data.get("prompts", []):
        candidates = prompt.get("candidates", [])
        for candidate_a in candidates:
            for candidate_b in candidates:
                if candidate_a["id"] == candidate_b["id"]:
                    continue
                rows.append(
                    {
                        "pair_id": f"{prompt['id']}:{candidate_a['id']}:{candidate_b['id']}",
                        "prompt": prompt["prompt"],
                        "image_a": str(PROJECT / candidate_a["asset"]),
                        "image_b": str(PROJECT / candidate_b["asset"]),
                    }
                )
    return pd.DataFrame(rows)


def attach_ordered_outputs(data: dict[str, Any], scored: pd.DataFrame) -> None:
    scored = scored.copy()
    parts = scored["pair_id"].astype(str).str.split(":", expand=True)
    scored["prompt_id"] = parts[0]
    scored["candidate_a"] = parts[1]
    scored["candidate_b"] = parts[2]
    dimensions = [
        column.removeprefix("prob_a_wins_")
        for column in scored.columns
        if column.startswith("prob_a_wins_")
    ]

    for prompt in data.get("prompts", []):
        candidate_ids = [candidate["id"] for candidate in prompt.get("candidates", [])]
        rows = scored[scored["prompt_id"] == prompt["id"]]
        ordered_scores: list[dict[str, Any]] = []
        by_candidate = {
            candidate_id: {dimension: [] for dimension in dimensions}
            for candidate_id in candidate_ids
        }

        for _, row in rows.iterrows():
            item: dict[str, Any] = {
                "candidate_a": row["candidate_a"],
                "candidate_b": row["candidate_b"],
                "probabilities": {},
            }
            for dimension in dimensions:
                probability = float(row[f"prob_a_wins_{dimension}"])
                item["probabilities"][dimension] = probability
                by_candidate[row["candidate_a"]][dimension].append(probability)
            ordered_scores.append(item)

        prompt["taste_ordered_pair_scores"] = ordered_scores

        for candidate in prompt.get("candidates", []):
            scores = {
                dimension: float(sum(values) / len(values)) if values else None
                for dimension, values in by_candidate[candidate["id"]].items()
            }
            valid_scores = [value for value in scores.values() if value is not None]
            candidate["taste_scores"] = scores
            candidate["model_output_scores"] = scores
            candidate["taste_overall"] = float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
            candidate["model_output_overall"] = candidate["taste_overall"]
            candidate["model_output_n_pairs"] = len(candidate_ids) - 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    df = build_ordered_pair_frame(data)
    scorer = load_scorer(args.checkpoint)
    scored = scorer.score_pairs(df, image_dir=None, batch_size=args.batch_size)
    attach_ordered_outputs(data, scored)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"scored ordered pairs: {len(scored)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
