from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
DATA_PATH = ROOT / "data.json"
INPUT_CSV = ROOT / "taste_pairs.csv"
SCORED_CSV = ROOT / "taste_pairs_scored.csv"
HF_REPO_ID = "purvanshi/TASTE"

SELECTED_PROMPTS = [
    {
        "prompt_id": 209,
        "title": "Travel mood board",
        "focus_dimension": "preference",
    },
    {
        "prompt_id": 315,
        "title": "Gardening Q&A post",
        "focus_dimension": "typography",
    },
    {
        "prompt_id": 628,
        "title": "Travel editorial layout",
        "focus_dimension": "spatial_accuracy",
    },
]

DIMENSION_LABELS = {
    "color_accuracy": "Color accuracy",
    "color_harmony": "Color harmony",
    "mood_and_color_tone": "Mood & color tone",
    "preference": "Preference",
    "spatial_accuracy": "Spatial accuracy",
    "typography": "Typography",
    "visual_hierarchy": "Visual hierarchy",
}


def short_prompt(text: str) -> str:
    text = " ".join(str(text).split())
    if text.startswith("User Intent: ** "):
        text = text.removeprefix("User Intent: ** ")
    return text


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = pd.read_parquet(f"hf://datasets/{HF_REPO_ID}/prompts.parquet")
    assets = pd.read_parquet(f"hf://datasets/{HF_REPO_ID}/assets.parquet")
    rankings = pd.read_parquet(f"hf://datasets/{HF_REPO_ID}/rankings.parquet")
    return prompts, assets, rankings


def copy_image(image_path: str, target: Path) -> None:
    source = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=image_path,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(source).read_bytes())


def human_pair_records(rank_rows: pd.DataFrame, candidate_ids: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for asset_a, asset_b in combinations(candidate_ids, 2):
        votes_a = 0
        votes_b = 0
        for _, evaluator_rows in rank_rows.groupby("evaluator_id"):
            ranks = evaluator_rows.set_index("asset_id")["rank"].to_dict()
            if ranks[asset_a] < ranks[asset_b]:
                votes_a += 1
            elif ranks[asset_b] < ranks[asset_a]:
                votes_b += 1
        rows.append(
            {
                "asset_id_a": int(asset_a),
                "asset_id_b": int(asset_b),
                "votes_a": votes_a,
                "votes_b": votes_b,
                "majority_winner": int(asset_a if votes_a >= votes_b else asset_b),
            }
        )
    return rows


def build_pair_csv(prompt_entries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prompt in prompt_entries:
        candidates = prompt["candidates"]
        for left, right in combinations(candidates, 2):
            rows.append(
                {
                    "pair_id": f"{prompt['id']}:{left['id']}:{right['id']}",
                    "prompt": prompt["prompt"],
                    "image_a": left["asset"],
                    "image_b": right["asset"],
                    "model_a": left["model"],
                    "model_b": right["model"],
                    "prompt_id": prompt["id"],
                    "candidate_a": left["id"],
                    "candidate_b": right["id"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(INPUT_CSV, index=False)
    return df


def run_taste_scorer(checkpoint: Path) -> pd.DataFrame | None:
    scorer_src = Path("/home/ubuntu/taste/taste-scorer/src")
    if str(scorer_src) not in sys.path:
        sys.path.insert(0, str(scorer_src))
    try:
        from taste_scorer import PreferenceScorer
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping TASTE scoring: could not import taste_scorer: {exc}")
        return None

    df = pd.read_csv(INPUT_CSV)
    scorer = PreferenceScorer.from_checkpoint(checkpoint, device="cuda")
    scored = scorer.score_pairs(df, image_dir=ROOT, batch_size=16)
    scored.to_csv(SCORED_CSV, index=False)
    return scored


def attach_taste_scores(prompt_entries: list[dict[str, Any]], scored: pd.DataFrame | None) -> None:
    if scored is None:
        return
    scored = scored.copy()
    pair_parts = scored["pair_id"].astype(str).str.split(":", expand=True)
    scored["prompt_id"] = pair_parts[0]
    scored["candidate_a"] = pair_parts[1]
    scored["candidate_b"] = pair_parts[2]
    dimensions = [
        column.removeprefix("prob_a_wins_")
        for column in scored.columns
        if column.startswith("prob_a_wins_")
    ]
    for prompt in prompt_entries:
        candidate_ids = [candidate["id"] for candidate in prompt["candidates"]]
        by_candidate = {
            candidate_id: {dimension: [] for dimension in dimensions}
            for candidate_id in candidate_ids
        }
        pair_scores = []
        prompt_rows = scored[scored["prompt_id"] == prompt["id"]]
        for _, row in prompt_rows.iterrows():
            item: dict[str, Any] = {
                "candidate_a": row["candidate_a"],
                "candidate_b": row["candidate_b"],
                "probabilities": {},
            }
            for dimension in dimensions:
                prob_a = float(row[f"prob_a_wins_{dimension}"])
                item["probabilities"][dimension] = prob_a
                by_candidate[row["candidate_a"]][dimension].append(prob_a)
                by_candidate[row["candidate_b"]][dimension].append(1.0 - prob_a)
            pair_scores.append(item)

        for candidate in prompt["candidates"]:
            candidate["taste_scores"] = {
                dimension: float(sum(values) / len(values)) if values else None
                for dimension, values in by_candidate[candidate["id"]].items()
            }
            candidate["taste_overall"] = float(
                sum(value for value in candidate["taste_scores"].values() if value is not None)
                / len([value for value in candidate["taste_scores"].values() if value is not None])
            )

        prompt["taste_pair_scores"] = pair_scores
        prompt["taste_dimensions"] = dimensions
        prompt["taste_rankings"] = {}
        for dimension in dimensions:
            ranked = sorted(
                prompt["candidates"],
                key=lambda candidate: candidate["taste_scores"].get(dimension) or 0.0,
                reverse=True,
            )
            prompt["taste_rankings"][dimension] = [
                {
                    "id": candidate["id"],
                    "label": candidate["label"],
                    "score": candidate["taste_scores"][dimension],
                }
                for candidate in ranked
            ]


def build_snapshot(score: bool, checkpoint: Path) -> dict[str, Any]:
    prompts, assets, rankings = load_tables()
    prompt_entries: list[dict[str, Any]] = []

    for selected in SELECTED_PROMPTS:
        prompt_id = selected["prompt_id"]
        prompt_row = prompts[prompts["prompt_id"] == prompt_id].iloc[0]
        rank_rows = rankings[rankings["prompt_id"] == prompt_id].copy()
        asset_ids = sorted(int(asset_id) for asset_id in rank_rows["asset_id"].unique())
        asset_rows = assets[assets["asset_id"].isin(asset_ids)].copy()

        mean_ranks = rank_rows.groupby("asset_id")["rank"].mean().to_dict()
        first_place_votes = (
            rank_rows[rank_rows["rank"] == 1].groupby("asset_id")["rank"].count().to_dict()
        )
        candidates: list[dict[str, Any]] = []
        ordered_assets = sorted(asset_ids, key=lambda asset_id: (mean_ranks[asset_id], asset_id))
        for index, asset_id in enumerate(ordered_assets, start=1):
            asset = asset_rows[asset_rows["asset_id"] == asset_id].iloc[0]
            ext = Path(asset["image_path"]).suffix.lower()
            target = ASSETS_DIR / str(prompt_id) / f"candidate-{index}{ext}"
            copy_image(str(asset["image_path"]), target)
            candidates.append(
                {
                    "id": f"c{index}",
                    "asset_id": int(asset_id),
                    "label": f"Candidate {index}",
                    "model": str(asset["model"]),
                    "asset": str(target.relative_to(ROOT)),
                    "human_mean_rank": float(mean_ranks[asset_id]),
                    "human_first_place_votes": int(first_place_votes.get(asset_id, 0)),
                }
            )

        prompt_entries.append(
            {
                "id": str(prompt_id),
                "title": selected["title"],
                "track": str(prompt_row["track"]),
                "dimension": str(prompt_row["dimension"]),
                "focus_dimension": selected["focus_dimension"],
                "dimension_label": DIMENSION_LABELS.get(selected["focus_dimension"], selected["focus_dimension"]),
                "prompt": short_prompt(str(prompt_row["prompt_text"])),
                "candidates": candidates,
                "human_pair_votes": human_pair_records(rank_rows, ordered_assets),
            }
        )

    build_pair_csv(prompt_entries)
    scored = run_taste_scorer(checkpoint) if score else None
    attach_taste_scores(prompt_entries, scored)

    data = {
        "title": "TASTE Score Samples",
        "source": {
            "dataset": "purvanshi/TASTE",
            "split": "train",
            "repo": "https://github.com/purvanshi-lica/taste",
        },
        "dimension_labels": DIMENSION_LABELS,
        "summary": {
            "prompts": len(prompt_entries),
            "candidates": sum(len(prompt["candidates"]) for prompt in prompt_entries),
            "pairs": sum(len(prompt["candidates"]) * (len(prompt["candidates"]) - 1) // 2 for prompt in prompt_entries),
            "taste_scored": scored is not None,
        },
        "prompts": prompt_entries,
    }
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--checkpoint", default="/home/ubuntu/TASTE_Checkpoint")
    args = parser.parse_args()
    data = build_snapshot(score=args.score, checkpoint=Path(args.checkpoint))
    print(json.dumps(data["summary"], indent=2))
    print(f"Wrote {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
