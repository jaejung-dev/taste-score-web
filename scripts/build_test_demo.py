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


def resolve_hf_image(image_path: str) -> str:
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        filename=image_path,
    )


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


def group_candidate_rankings(group: pd.DataFrame) -> pd.DataFrame:
    long_rows: list[dict[str, Any]] = []
    for _, row in group.iterrows():
        long_rows.append({"filename": str(row["image_a"]), "mean_rank": float(row["mean_rank_a"])})
        long_rows.append({"filename": str(row["image_b"]), "mean_rank": float(row["mean_rank_b"])})
    return (
        pd.DataFrame(long_rows)
        .groupby("filename")
        .agg(human_mean_rank=("mean_rank", "mean"))
        .reset_index()
        .sort_values(["human_mean_rank", "filename"])
    )


def select_model_aware_test_groups(
    battles: pd.DataFrame,
    assets: pd.DataFrame,
    checkpoint: Path,
    max_per_dimension: int,
) -> pd.DataFrame:
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
    stat_lookup = {
        (str(row.dimension), int(row.prompt_id), str(row.scene_id)): row
        for row in stats.itertuples(index=False)
    }

    asset_by_filename = assets.set_index("filename")
    resolved_paths = {
        filename: resolve_hf_image(str(row.image_path))
        for filename, row in asset_by_filename.iterrows()
    }

    ordered_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for group_index, (group_key, group) in enumerate(battles.groupby(group_cols), start=1):
        dimension, prompt_id, scene_id = group_key
        ranked = group_candidate_rankings(group)
        filenames = ranked["filename"].astype(str).tolist()
        prompt = str(group["prompt"].iloc[0])
        for left in filenames:
            for right in filenames:
                if left == right:
                    continue
                pair_id = f"select-{group_index:04d}-{len(ordered_rows):05d}"
                ordered_rows.append(
                    {
                        "pair_id": pair_id,
                        "prompt": prompt,
                        "image_a": resolved_paths[left],
                        "image_b": resolved_paths[right],
                    }
                )
                meta_rows.append(
                    {
                        "pair_id": pair_id,
                        "dimension": str(dimension),
                        "prompt_id": int(prompt_id),
                        "scene_id": str(scene_id),
                        "image_a": left,
                        "image_b": right,
                    }
                )

    scorer = load_taste_scorer(checkpoint)
    scored = scorer.score_pairs(pd.DataFrame(ordered_rows), image_dir=None, batch_size=32)
    scored = scored.merge(pd.DataFrame(meta_rows), on="pair_id", how="left", suffixes=("", "_meta"))

    selection_rows: list[dict[str, Any]] = []
    for group_key, group in battles.groupby(group_cols):
        dimension, prompt_id, scene_id = group_key
        ranked = group_candidate_rankings(group)
        human_top = str(ranked.iloc[0]["filename"])
        focus = str(dimension)
        score_rows = scored[
            (scored["dimension"] == focus)
            & (scored["prompt_id"] == int(prompt_id))
            & (scored["scene_id"] == str(scene_id))
        ]
        by_image: dict[str, list[float]] = {str(filename): [] for filename in ranked["filename"]}
        for _, row in score_rows.iterrows():
            by_image[str(row["image_a_meta"])].append(float(row[f"prob_a_wins_{focus}"]))
        taste_scores = {
            filename: float(sum(values) / len(values)) if values else 0.0
            for filename, values in by_image.items()
        }
        taste_ranked = sorted(taste_scores.items(), key=lambda item: item[1], reverse=True)
        taste_top, taste_top_score = taste_ranked[0]
        taste_second_score = taste_ranked[1][1] if len(taste_ranked) > 1 else 0.0
        stat = stat_lookup[(focus, int(prompt_id), str(scene_id))]
        selection_rows.append(
            {
                **stat._asdict(),
                "human_top": human_top,
                "taste_top": taste_top,
                "taste_top_score": taste_top_score,
                "taste_margin": float(taste_top_score - taste_second_score),
                "taste_matches_human": bool(taste_top == human_top),
            }
        )

    selection = pd.DataFrame(selection_rows)
    picked: list[pd.DataFrame] = []
    used_scenes: set[str] = set()
    for dimension in DIMENSION_ORDER:
        candidates = selection[selection["dimension"] == dimension].sort_values(
            ["taste_matches_human", "mean_agreement", "taste_margin", "unanimous_rate", "prompt_id"],
            ascending=[False, False, False, False, True],
        )
        rows = []
        for _, row in candidates.iterrows():
            if len(rows) >= max_per_dimension:
                break
            if row["scene_id"] in used_scenes and len(candidates) > max_per_dimension:
                continue
            rows.append(row)
            used_scenes.add(str(row["scene_id"]))
        if len(rows) < max_per_dimension:
            for _, row in candidates.iterrows():
                if len(rows) >= max_per_dimension:
                    break
                if any(
                    (picked_row["dimension"], picked_row["prompt_id"], picked_row["scene_id"])
                    == (row["dimension"], row["prompt_id"], row["scene_id"])
                    for picked_row in rows
                ):
                    continue
                rows.append(row)
        picked.append(pd.DataFrame(rows))
    return pd.concat(picked, ignore_index=True).sort_values(["dimension_order", "prompt_id"])


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


def load_taste_scorer(checkpoint: Path):
    scorer_src = Path("/home/ubuntu/taste/taste-scorer/src")
    if str(scorer_src) not in sys.path:
        sys.path.insert(0, str(scorer_src))
    from taste_scorer import PreferenceScorer

    return PreferenceScorer.from_checkpoint(checkpoint, device="cuda")


def run_taste_scorer(checkpoint: Path) -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    scorer = load_taste_scorer(checkpoint)
    scored = scorer.score_pairs(df, image_dir=ROOT, batch_size=16)
    scored.to_csv(SCORED_CSV, index=False)
    return scored


def score_dataframe(df: pd.DataFrame, checkpoint: Path, output_csv: Path | None = None) -> pd.DataFrame:
    scorer = load_taste_scorer(checkpoint)
    scored = scorer.score_pairs(df, image_dir=None, batch_size=32)
    if output_csv is not None:
        scored.to_csv(output_csv, index=False)
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


def pair_key_frame(battles: pd.DataFrame) -> pd.Series:
    return battles.apply(
        lambda row: (
            row["dimension"],
            int(row["prompt_id"]),
            str(row["scene_id"]),
            tuple(sorted([str(row["image_a"]), str(row["image_b"])])),
        ),
        axis=1,
    )


def full_eval_input(battles: pd.DataFrame, assets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    asset_by_filename = assets.set_index("filename")
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    work = battles.copy()
    work["_pair_key"] = pair_key_frame(work)

    for index, (pair_key, pair_rows) in enumerate(work.groupby("_pair_key"), start=1):
        dimension, prompt_id, scene_id, image_pair = pair_key
        image_a, image_b = image_pair
        first = pair_rows.iloc[0]
        votes_a = 0
        for _, row in pair_rows.iterrows():
            winner_file = row["image_a"] if row["winner"] == "A" else row["image_b"]
            if str(winner_file) == image_a:
                votes_a += 1
        n_votes = int(len(pair_rows))
        image_path_a = str(asset_by_filename.loc[image_a, "image_path"])
        image_path_b = str(asset_by_filename.loc[image_b, "image_path"])
        pair_id = f"eval-{index:04d}"
        rows.append(
            {
                "pair_id": pair_id,
                "prompt": str(first["prompt"]),
                "image_a": resolve_hf_image(image_path_a),
                "image_b": resolve_hf_image(image_path_b),
                "model_a": str(asset_by_filename.loc[image_a, "model"]),
                "model_b": str(asset_by_filename.loc[image_b, "model"]),
            }
        )
        meta_rows.append(
            {
                "pair_id": pair_id,
                "dimension": str(dimension),
                "prompt_id": int(prompt_id),
                "scene_id": str(scene_id),
                "image_a_filename": image_a,
                "image_b_filename": image_b,
                "human_votes_a": votes_a,
                "human_votes_b": n_votes - votes_a,
                "human_vote_share_a": float(votes_a / n_votes),
                "human_majority_a": bool(votes_a > n_votes / 2),
                "human_agreement": float(max(votes_a, n_votes - votes_a) / n_votes),
                "n_votes": n_votes,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def corr_or_none(left: pd.Series, right: pd.Series, method: str = "pearson") -> float | None:
    value = left.astype(float).corr(right.astype(float), method=method)
    if pd.isna(value):
        return None
    return float(value)


def evaluate_full_test(battles: pd.DataFrame, assets: pd.DataFrame, checkpoint: Path) -> dict[str, Any]:
    eval_input, eval_meta = full_eval_input(battles, assets)
    scored = score_dataframe(eval_input, checkpoint=checkpoint)
    scored = scored.merge(eval_meta, on="pair_id", how="left")
    scored["model_prob_a"] = scored.apply(
        lambda row: float(row[f"prob_a_wins_{row['dimension']}"]),
        axis=1,
    )
    scored["model_majority_a"] = scored["model_prob_a"] >= 0.5
    scored["pairwise_correct"] = scored["model_majority_a"] == scored["human_majority_a"]

    prompt_rows: list[dict[str, Any]] = []
    image_rank_rows: list[dict[str, Any]] = []
    for group_key, group in battles.groupby(["dimension", "prompt_id", "scene_id"]):
        dimension, prompt_id, scene_id = group_key
        score_values: dict[str, list[float]] = {}
        human_ranks: dict[str, list[float]] = {}
        for _, row in group.iterrows():
            human_ranks.setdefault(str(row["image_a"]), []).append(float(row["mean_rank_a"]))
            human_ranks.setdefault(str(row["image_b"]), []).append(float(row["mean_rank_b"]))
        for _, row in scored[
            (scored["dimension"] == dimension)
            & (scored["prompt_id"] == prompt_id)
            & (scored["scene_id"] == scene_id)
        ].iterrows():
            score_values.setdefault(row["image_a_filename"], []).append(float(row["model_prob_a"]))
            score_values.setdefault(row["image_b_filename"], []).append(1.0 - float(row["model_prob_a"]))
        candidates = sorted(human_ranks)
        if not candidates:
            continue
        summary = pd.DataFrame(
            {
                "filename": candidates,
                "human_rank": [sum(human_ranks[c]) / len(human_ranks[c]) for c in candidates],
                "model_score": [sum(score_values[c]) / len(score_values[c]) for c in candidates],
            }
        )
        human_best = str(summary.sort_values(["human_rank", "filename"]).iloc[0]["filename"])
        model_best = str(summary.sort_values(["model_score", "filename"], ascending=[False, True]).iloc[0]["filename"])
        spearman = corr_or_none(summary["model_score"], -summary["human_rank"], method="spearman")
        prompt_rows.append(
            {
                "dimension": str(dimension),
                "prompt_id": int(prompt_id),
                "scene_id": str(scene_id),
                "human_best": human_best,
                "model_best": model_best,
                "top1_correct": human_best == model_best,
                "rank_spearman": spearman,
            }
        )
        for row in summary.itertuples(index=False):
            image_rank_rows.append(
                {
                    "dimension": str(dimension),
                    "prompt_id": int(prompt_id),
                    "filename": str(row.filename),
                    "human_rank": float(row.human_rank),
                    "model_score": float(row.model_score),
                }
            )

    prompt_eval = pd.DataFrame(prompt_rows)
    image_eval = pd.DataFrame(image_rank_rows)
    by_dimension: list[dict[str, Any]] = []
    for dimension in DIMENSION_ORDER:
        pair_dim = scored[scored["dimension"] == dimension]
        prompt_dim = prompt_eval[prompt_eval["dimension"] == dimension]
        image_dim = image_eval[image_eval["dimension"] == dimension]
        if pair_dim.empty:
            continue
        by_dimension.append(
            {
                "dimension": dimension,
                "label": DIMENSION_LABELS.get(dimension, dimension),
                "n_pairs": int(len(pair_dim)),
                "pairwise_accuracy": float(pair_dim["pairwise_correct"].mean()),
                "vote_share_spearman": corr_or_none(
                    pair_dim["model_prob_a"], pair_dim["human_vote_share_a"], method="spearman"
                ),
                "vote_share_pearson": corr_or_none(
                    pair_dim["model_prob_a"], pair_dim["human_vote_share_a"], method="pearson"
                ),
                "prompt_top1_accuracy": float(prompt_dim["top1_correct"].mean()) if not prompt_dim.empty else None,
                "image_rank_spearman": corr_or_none(
                    image_dim["model_score"], -image_dim["human_rank"], method="spearman"
                ) if not image_dim.empty else None,
            }
        )

    overview = {
        "n_pairs": int(len(scored)),
        "n_prompt_groups": int(len(prompt_eval)),
        "pairwise_accuracy": float(scored["pairwise_correct"].mean()),
        "vote_share_spearman": corr_or_none(
            scored["model_prob_a"], scored["human_vote_share_a"], method="spearman"
        ),
        "vote_share_pearson": corr_or_none(
            scored["model_prob_a"], scored["human_vote_share_a"], method="pearson"
        ),
        "prompt_top1_accuracy": float(prompt_eval["top1_correct"].mean()),
        "prompt_rank_spearman_mean": float(prompt_eval["rank_spearman"].dropna().mean()),
        "image_rank_spearman": corr_or_none(
            image_eval["model_score"], -image_eval["human_rank"], method="spearman"
        ),
    }
    return {
        "overview": overview,
        "by_dimension": by_dimension,
        "notes": [
            "Pairwise accuracy compares P(A wins B) >= 0.5 with the 5-rater human majority winner.",
            "Vote-share correlation compares model P(A wins B) with the fraction of humans choosing A.",
            "Top-1 accuracy aggregates pairwise probabilities within each 4-image prompt and compares the model-best image with the lowest human mean rank.",
        ],
    }


def build_snapshot(battle_csv: Path, checkpoint: Path, max_per_dimension: int) -> dict[str, Any]:
    battles = pd.read_csv(battle_csv)
    assets = load_assets()
    if ASSETS_DIR.exists():
        shutil.rmtree(ASSETS_DIR)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    selected = select_model_aware_test_groups(
        battles,
        assets=assets,
        checkpoint=checkpoint,
        max_per_dimension=max_per_dimension,
    )
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
    evaluation = evaluate_full_test(battles, assets, checkpoint)

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
        "score_explanation": "TASTE evaluates generated design images against human visual preferences across quality dimensions.",
        "evaluation": evaluation,
        "taste_dimensions": dimensions,
        "prompts": prompt_entries,
    }
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--battle-csv", default="/home/ubuntu/battles_test.csv")
    parser.add_argument("--checkpoint", default="/home/ubuntu/TASTE_Checkpoint")
    parser.add_argument("--max-per-dimension", type=int, default=3)
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
