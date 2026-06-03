from __future__ import annotations

import argparse
import json
import shutil
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

DIMENSION_ORDER = [
    "preference",
    "typography",
    "visual_hierarchy",
    "color_harmony",
    "mood_and_color_tone",
    "spatial_accuracy",
    "color_accuracy",
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


def load_assets() -> pd.DataFrame:
    assets = pd.read_parquet(f"hf://datasets/{HF_REPO_ID}/assets.parquet")
    assets = assets.copy()
    assets["filename"] = assets["image_path"].map(lambda value: Path(str(value)).name)
    return assets


def copy_image(image_path: str, target: Path) -> None:
    source = hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=image_path,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(source).read_bytes())


def select_test_groups(battles: pd.DataFrame, max_per_dimension: int) -> pd.DataFrame:
    group_cols = ["dimension", "prompt_id", "scene_id"]
    stats = (
        battles.groupby(group_cols)
        .agg(
            prompt=("prompt", "first"),
            prompt_template=("prompt_template", "first"),
            mean_agreement=("agreement", "mean"),
            unanimous_rate=("agreement_bucket", lambda s: float((s == "unanimous").mean())),
            n_battles=("winner", "size"),
        )
        .reset_index()
    )
    stats["dimension_order"] = stats["dimension"].map(
        {dimension: index for index, dimension in enumerate(DIMENSION_ORDER)}
    )
    selected: list[pd.DataFrame] = []
    used_scenes: set[str] = set()
    for dimension in DIMENSION_ORDER:
        candidates = stats[stats["dimension"] == dimension].sort_values(
            ["mean_agreement", "unanimous_rate", "prompt_id"],
            ascending=[False, False, True],
        )
        picked_rows = []
        for _, row in candidates.iterrows():
            if len(picked_rows) >= max_per_dimension:
                break
            if row["scene_id"] in used_scenes and len(candidates) > max_per_dimension:
                continue
            picked_rows.append(row)
            used_scenes.add(str(row["scene_id"]))
        if not picked_rows:
            picked_rows = [candidates.iloc[0]]
        selected.append(pd.DataFrame(picked_rows))
    return pd.concat(selected, ignore_index=True).sort_values("dimension_order")


def candidate_records(group: pd.DataFrame, assets: pd.DataFrame) -> list[dict[str, Any]]:
    long_rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        long_rows.append(
            {
                "filename": row["image_a"],
                "url": row["image_url_a"],
                "model": row["model_a"],
                "rank": row["rank_a"],
                "mean_rank": row["mean_rank_a"],
                "evaluator": row["evaluator"],
            }
        )
        long_rows.append(
            {
                "filename": row["image_b"],
                "url": row["image_url_b"],
                "model": row["model_b"],
                "rank": row["rank_b"],
                "mean_rank": row["mean_rank_b"],
                "evaluator": row["evaluator"],
            }
        )

    long_df = pd.DataFrame(long_rows).drop_duplicates(["filename", "evaluator"])
    by_file = (
        long_df.groupby("filename")
        .agg(
            image_url=("url", "first"),
            model=("model", "first"),
            human_mean_rank=("mean_rank", "mean"),
            human_first_place_votes=("rank", lambda s: int((s == 1).sum())),
        )
        .reset_index()
        .sort_values(["human_mean_rank", "filename"])
    )

    candidates: list[dict[str, Any]] = []
    for index, row in enumerate(by_file.itertuples(index=False), start=1):
        asset_row = assets[assets["filename"] == row.filename].iloc[0]
        source_path = str(asset_row.image_path)
        ext = Path(source_path).suffix.lower()
        prompt_dir = f"{int(group['prompt_id'].iloc[0]):04d}-{group['dimension'].iloc[0]}"
        target = ASSETS_DIR / prompt_dir / f"candidate-{index}{ext}"
        copy_image(source_path, target)
        candidates.append(
            {
                "id": f"c{index}",
                "label": f"Candidate {index}",
                "filename": str(row.filename),
                "asset_id": int(asset_row.asset_id),
                "model": str(row.model),
                "image_url": str(row.image_url),
                "asset": str(target.relative_to(ROOT)),
                "human_mean_rank": float(row.human_mean_rank),
                "human_first_place_votes": int(row.human_first_place_votes),
            }
        )
    return candidates


def human_pair_votes(group: pd.DataFrame, filename_to_id: dict[str, str]) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    for pair_key, pair_df in group.groupby(
        group.apply(lambda r: tuple(sorted([r["image_a"], r["image_b"]])), axis=1)
    ):
        left_file, right_file = pair_key
        left_id = filename_to_id[left_file]
        right_id = filename_to_id[right_file]
        left_votes = 0
        right_votes = 0
        for _, row in pair_df.iterrows():
            winner_file = row["image_a"] if row["winner"] == "A" else row["image_b"]
            if winner_file == left_file:
                left_votes += 1
            else:
                right_votes += 1
        pair_rows.append(
            {
                "candidate_a": left_id,
                "candidate_b": right_id,
                "votes_a": left_votes,
                "votes_b": right_votes,
                "agreement": float(max(left_votes, right_votes) / (left_votes + right_votes)),
                "majority_winner": left_id if left_votes >= right_votes else right_id,
            }
        )
    return sorted(pair_rows, key=lambda row: (row["candidate_a"], row["candidate_b"]))


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
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(INPUT_CSV, index=False)
    return df


def run_taste_scorer(checkpoint: Path) -> pd.DataFrame:
    scorer_src = Path("/home/ubuntu/taste/taste-scorer/src")
    if str(scorer_src) not in sys.path:
        sys.path.insert(0, str(scorer_src))
    from taste_scorer import PreferenceScorer

    df = pd.read_csv(INPUT_CSV)
    scorer = PreferenceScorer.from_checkpoint(checkpoint, device="cuda")
    scored = scorer.score_pairs(df, image_dir=ROOT, batch_size=16)
    scored.to_csv(SCORED_CSV, index=False)
    return scored


def attach_model_outputs(prompt_entries: list[dict[str, Any]], scored: pd.DataFrame) -> list[str]:
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
            model_output_scores = {
                dimension: float(sum(values) / len(values)) if values else None
                for dimension, values in by_candidate[candidate["id"]].items()
            }
            valid_scores = [
                value for value in model_output_scores.values() if value is not None
            ]
            candidate["model_output_scores"] = model_output_scores
            candidate["taste_scores"] = model_output_scores
            candidate["model_output_overall"] = (
                float(sum(valid_scores) / len(valid_scores)) if valid_scores else None
            )
            candidate["taste_overall"] = candidate["model_output_overall"]
            candidate["model_output_n_pairs"] = len(
                by_candidate[candidate["id"]][prompt["focus_dimension"]]
            )

        prompt["taste_pair_scores"] = pair_scores
        prompt["taste_dimensions"] = dimensions
        prompt["model_output_note"] = (
            "Per-image scores are aggregated from the pairwise model output: "
            "mean P(candidate wins) against the other candidates in this prompt."
        )
        prompt["taste_rankings"] = {}
        for dimension in dimensions:
            ranked = sorted(
                prompt["candidates"],
                key=lambda candidate: candidate["model_output_scores"].get(dimension) or 0.0,
                reverse=True,
            )
            prompt["taste_rankings"][dimension] = [
                {
                    "id": candidate["id"],
                    "label": candidate["label"],
                    "score": candidate["model_output_scores"][dimension],
                }
                for candidate in ranked
            ]
    return dimensions


def build_snapshot(battle_csv: Path, checkpoint: Path, max_per_dimension: int) -> dict[str, Any]:
    battles = pd.read_csv(battle_csv)
    assets = load_assets()
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_test_groups(battles, max_per_dimension=max_per_dimension)
    prompt_entries: list[dict[str, Any]] = []
    for _, selected_row in selected.iterrows():
        group = battles[
            (battles["dimension"] == selected_row["dimension"])
            & (battles["prompt_id"] == selected_row["prompt_id"])
            & (battles["scene_id"] == selected_row["scene_id"])
        ].copy()
        candidates = candidate_records(group, assets)
        filename_to_id = {candidate["filename"]: candidate["id"] for candidate in candidates}
        dimension = str(selected_row["dimension"])
        prompt_entries.append(
            {
                "id": f"{int(selected_row['prompt_id'])}-{dimension}",
                "prompt_id": int(selected_row["prompt_id"]),
                "scene_id": str(selected_row["scene_id"]),
                "title": f"{DIMENSION_LABELS.get(dimension, dimension)} sample",
                "track": "test",
                "dimension": dimension,
                "focus_dimension": dimension,
                "dimension_label": DIMENSION_LABELS.get(dimension, dimension),
                "prompt_template": str(selected_row["prompt_template"]),
                "prompt": short_prompt(str(selected_row["prompt"])),
                "mean_human_agreement": float(selected_row["mean_agreement"]),
                "unanimous_rate": float(selected_row["unanimous_rate"]),
                "candidates": candidates,
                "human_pair_votes": human_pair_votes(group, filename_to_id),
            }
        )

    build_pair_csv(prompt_entries)
    scored = run_taste_scorer(checkpoint)
    dimensions = attach_model_outputs(prompt_entries, scored)

    data = {
        "title": "TASTE Test Score Samples",
        "source": {
            "dataset": "purvanshi/TASTE",
            "split": "test manifest from battles_test.csv",
            "battle_csv": battle_csv.name,
            "repo": "https://github.com/purvanshi-lica/taste",
        },
        "dimension_labels": DIMENSION_LABELS,
        "dimension_order": DIMENSION_ORDER,
        "summary": {
            "test_rows": int(len(battles)),
            "test_unique_pairs": int(
                battles.apply(
                    lambda r: (r["dimension"], r["prompt_id"], tuple(sorted([r["image_a"], r["image_b"]]))),
                    axis=1,
                ).nunique()
            ),
            "test_prompts": int(battles["prompt_id"].nunique()),
            "test_images": int(
                len(set(battles["image_a"].astype(str)).union(set(battles["image_b"].astype(str))))
            ),
            "selected_prompts": len(prompt_entries),
            "selected_candidates": sum(len(prompt["candidates"]) for prompt in prompt_entries),
            "selected_pairs": sum(len(prompt["taste_pair_scores"]) for prompt in prompt_entries),
            "taste_scored": True,
        },
        "score_explanation": (
            "TASTE is a pairwise preference scorer. Raw model outputs are "
            "P(image A wins over image B) for each dimension. The per-image "
            "score shown here is the mean win probability for that image "
            "against the other candidates in the same prompt."
        ),
        "taste_dimensions": dimensions,
        "prompts": prompt_entries,
    }
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battle-csv", default="/home/ubuntu/battles_test.csv")
    parser.add_argument("--checkpoint", default="/home/ubuntu/TASTE_Checkpoint")
    parser.add_argument("--max-per-dimension", type=int, default=1)
    args = parser.parse_args()
    data = build_snapshot(
        battle_csv=Path(args.battle_csv),
        checkpoint=Path(args.checkpoint),
        max_per_dimension=args.max_per_dimension,
    )
    print(json.dumps(data["summary"], indent=2))
    print(f"Wrote {DATA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
