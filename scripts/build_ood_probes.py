from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import cairosvg
import pandas as pd
import requests
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data.json"
OOD_ASSET_DIR = PROJECT / "assets" / "ood"
PROD_ENV_PATH = Path("/home/ubuntu/web-app/ml-engine/app/prod-env.yaml")
SVG_CANDIDATES = Path("/home/ubuntu/ml-platform/other-projects/lica-score/data/processed/svg_data_v1/candidates.jsonl")
IMSCORE_ENDPOINT = "https://model-wnpgy843.api.baseten.co/environments/production/predict"
TASTE_CHECKPOINT = Path("/home/ubuntu/TASTE_Checkpoint")

IMSCORE_ORDER = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"]
IMSCORE_LABELS = {
    "hpsv21": "HPS v2.1",
    "pickscore": "PickScore",
    "clipscore": "CLIPScore",
    "imagereward": "ImageReward",
    "laion_aesthetic": "LAION aesthetic",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: str, outline: str | None = None, width: int = 1) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def save_chart_assets() -> list[dict[str, Any]]:
    asset_dir = OOD_ASSET_DIR / "chart-infographic"
    asset_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("Executive dashboard", "#eef6ff", "#2454a6", "#29a36a"),
        ("Quarterly revenue", "#fff7ed", "#d97706", "#2563eb"),
        ("Market segments", "#f7fee7", "#65a30d", "#7c3aed"),
        ("Growth funnel", "#f8fafc", "#0f766e", "#f43f5e"),
    ]
    candidates = []
    for idx, (title, bg, primary, accent) in enumerate(specs, start=1):
        img = Image.new("RGB", (1024, 768), bg)
        draw = ImageDraw.Draw(img)
        draw.text((60, 44), title, fill="#172033", font=font(46, True))
        draw.text((62, 102), "OOD technical chart / infographic probe", fill="#64748b", font=font(22))
        for card_idx in range(3):
            x = 62 + card_idx * 300
            rounded_rect(draw, (x, 150, x + 250, 255), 24, "#ffffff", "#d8dee9", 2)
            draw.text((x + 22, 174), ["Revenue", "Users", "Margin"][card_idx], fill="#64748b", font=font(18, True))
            draw.text((x + 22, 205), ["$2.4M", "84K", "31%"][card_idx], fill="#172033", font=font(32, True))
        # Bar chart
        chart = (70, 315, 520, 675)
        rounded_rect(draw, chart, 28, "#ffffff", "#d8dee9", 2)
        draw.text((100, 340), "Monthly trend", fill="#172033", font=font(24, True))
        base_y = 628
        values = [140, 185, 110, 230, 260, 205]
        for i, value in enumerate(values):
            x0 = 115 + i * 62
            draw.rounded_rectangle((x0, base_y - value, x0 + 34, base_y), radius=10, fill=primary)
            draw.text((x0 - 2, base_y + 12), f"M{i+1}", fill="#64748b", font=font(14))
        # Line chart / diagram
        panel = (560, 315, 955, 675)
        rounded_rect(draw, panel, 28, "#ffffff", "#d8dee9", 2)
        draw.text((590, 340), "Signal flow", fill="#172033", font=font(24, True))
        points = [(610, 590), (680, 510), (750, 535), (820, 430), (905, 380)]
        draw.line(points, fill=accent, width=8, joint="curve")
        for x, y in points:
            draw.ellipse((x - 12, y - 12, x + 12, y + 12), fill="#ffffff", outline=accent, width=6)
        for i, label in enumerate(["Input", "Clean", "Score", "Rank"]):
            y = 420 + i * 46
            rounded_rect(draw, (590, y, 735, y + 30), 10, "#eef2ff" if i % 2 else "#ecfdf5")
            draw.text((606, y + 5), label, fill="#334155", font=font(15, True))
        path = asset_dir / f"candidate-{idx}.png"
        img.save(path)
        candidates.append({"id": f"c{idx}", "label": f"Candidate {idx}", "asset": str(path.relative_to(PROJECT))})
    return candidates


def save_mobile_assets() -> list[dict[str, Any]]:
    asset_dir = OOD_ASSET_DIR / "mobile-ui"
    asset_dir.mkdir(parents=True, exist_ok=True)
    themes = [
        ("Finance", "#0f172a", "#38bdf8", "#f8fafc"),
        ("Fitness", "#064e3b", "#34d399", "#f0fdf4"),
        ("Food", "#7f1d1d", "#fb923c", "#fff7ed"),
        ("Travel", "#312e81", "#a78bfa", "#faf5ff"),
    ]
    candidates = []
    for idx, (name, dark, accent, bg) in enumerate(themes, start=1):
        img = Image.new("RGB", (768, 1024), "#ece7dd")
        draw = ImageDraw.Draw(img)
        rounded_rect(draw, (150, 42, 618, 982), 58, "#111827")
        rounded_rect(draw, (174, 72, 594, 952), 44, bg)
        draw.rounded_rectangle((315, 88, 455, 100), radius=6, fill="#0f172a")
        draw.text((210, 135), f"{name} App", fill=dark, font=font(42, True))
        draw.text((212, 190), "Clean onboarding and dashboard UI", fill="#64748b", font=font(19))
        rounded_rect(draw, (214, 245, 554, 410), 34, dark)
        draw.text((244, 280), "Today", fill="#cbd5e1", font=font(20, True))
        draw.text((244, 318), "72%", fill="#ffffff", font=font(64, True))
        draw.arc((410, 275, 520, 385), start=10, end=320, fill=accent, width=13)
        for row in range(3):
            y = 455 + row * 105
            rounded_rect(draw, (214, y, 554, y + 78), 24, "#ffffff", "#e2e8f0", 2)
            draw.ellipse((240, y + 20, 278, y + 58), fill=accent)
            draw.text((300, y + 18), ["Overview", "Insights", "Next step"][row], fill="#172033", font=font(22, True))
            draw.text((300, y + 48), ["Track progress", "Compare trends", "Take action"][row], fill="#64748b", font=font(15))
        rounded_rect(draw, (214, 805, 554, 878), 26, accent)
        draw.text((310, 827), "Continue", fill="#ffffff", font=font(25, True))
        path = asset_dir / f"candidate-{idx}.png"
        img.save(path)
        candidates.append({"id": f"c{idx}", "label": f"Candidate {idx}", "asset": str(path.relative_to(PROJECT))})
    return candidates


def download_photo_assets() -> list[dict[str, Any]]:
    asset_dir = OOD_ASSET_DIR / "photorealistic"
    asset_dir.mkdir(parents=True, exist_ok=True)
    ids = [237, 1025, 1062, 1084]
    candidates = []
    for idx, image_id in enumerate(ids, start=1):
        path = asset_dir / f"candidate-{idx}.jpg"
        url = f"https://picsum.photos/id/{image_id}/1024/1024"
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        except Exception:
            img = Image.new("RGB", (1024, 1024), "#d7d2c5")
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 1024, 580), fill="#93c5fd")
            draw.rectangle((0, 580, 1024, 1024), fill="#86efac")
            draw.ellipse((300, 260, 760, 760), fill="#f5deb3", outline="#7c2d12", width=12)
            draw.text((280, 815), f"Photo fallback {idx}", fill="#172033", font=font(42, True))
            img.save(path)
        candidates.append({"id": f"c{idx}", "label": f"Candidate {idx}", "asset": str(path.relative_to(PROJECT))})
    return candidates


def svg_complex_candidates(group_id: str = "2e9fa887-7a3b-49d5-a23f-91b7176270c1") -> tuple[str, list[dict[str, Any]]]:
    asset_dir = OOD_ASSET_DIR / "svg-complex"
    asset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for line in SVG_CANDIDATES.read_text().splitlines():
        row = json.loads(line)
        if row.get("group_id") == group_id and row.get("bucket") == "complex":
            rows.append(row)
    order = ["gt", "gpt-5.2", "claude", "gemini"]
    by_source = {row["source"]: row for row in rows}
    selected = [by_source[source] for source in order if source in by_source]
    if len(selected) != 4:
        raise RuntimeError(f"Expected four SVG complex candidates for {group_id}, found {len(selected)}")
    prompt = selected[0]["prompt_text"]
    candidates = []
    for idx, row in enumerate(selected, start=1):
        png_path = asset_dir / f"candidate-{idx}.png"
        cairosvg.svg2png(url=row["svg_path"], write_to=str(png_path), output_width=1024, output_height=1024)
        candidates.append(
            {
                "id": f"c{idx}",
                "label": f"Candidate {idx}",
                "asset": str(png_path.relative_to(PROJECT)),
            }
        )
    return prompt, candidates


def load_prod_env(path: Path = PROD_ENV_PATH) -> None:
    if not path.exists():
        return
    data = yaml.safe_load(path.read_text()) or {}
    for key, value in (data.get("env_variables") or {}).items():
        if value is not None:
            os.environ.setdefault(key, str(value))


def image_data_uri(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=86, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def load_taste_scorer(checkpoint: Path = TASTE_CHECKPOINT):
    sys.path.insert(0, "/home/ubuntu/taste/taste-scorer/src")
    from taste_scorer import PreferenceScorer

    return PreferenceScorer.from_checkpoint(checkpoint, device="cuda")


def score_taste_preference(ood_prompts: list[dict[str, Any]]) -> None:
    rows = []
    for prompt in ood_prompts:
        for cand_a in prompt["candidates"]:
            for cand_b in prompt["candidates"]:
                if cand_a["id"] == cand_b["id"]:
                    continue
                rows.append(
                    {
                        "pair_id": f"{prompt['id']}:{cand_a['id']}:{cand_b['id']}",
                        "prompt": prompt["prompt"],
                        "image_a": str(PROJECT / cand_a["asset"]),
                        "image_b": str(PROJECT / cand_b["asset"]),
                    }
                )
    scorer = load_taste_scorer()
    scored = scorer.score_pairs(pd.DataFrame(rows), image_dir=None, batch_size=32)
    for prompt in ood_prompts:
        by_candidate = {candidate["id"]: [] for candidate in prompt["candidates"]}
        pair_scores = []
        prompt_rows = scored[scored["pair_id"].astype(str).str.startswith(f"{prompt['id']}:")]
        for _, row in prompt_rows.iterrows():
            _, a_id, b_id = str(row["pair_id"]).split(":")
            probability = float(row["prob_a_wins_preference"])
            by_candidate[a_id].append(probability)
            pair_scores.append({"candidate_a": a_id, "candidate_b": b_id, "probabilities": {"preference": probability}})
        prompt["taste_ordered_pair_scores"] = pair_scores
        for candidate in prompt["candidates"]:
            values = by_candidate[candidate["id"]]
            score = float(sum(values) / len(values)) if values else None
            candidate["taste_scores"] = {"preference": score}


def call_imscore(prompt_text: str, candidates: list[dict[str, Any]], api_key: str, timeout: float) -> dict[str, dict[str, float]]:
    response = requests.post(
        IMSCORE_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "prompt": prompt_text,
            "metrics": IMSCORE_ORDER,
            "candidates": [
                {"id": candidate["id"], "image": image_data_uri(PROJECT / candidate["asset"])}
                for candidate in candidates
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    raw = response.json().get("scores", {})
    return {
        candidate_id: {metric: float(value) for metric, value in scores.items()}
        for candidate_id, scores in raw.items()
    }


def score_imscore(ood_prompts: list[dict[str, Any]], api_key: str, timeout: float) -> None:
    for prompt in ood_prompts:
        print(f"imscore {prompt['id']} group", flush=True)
        try:
            scores = call_imscore(prompt["prompt"], prompt["candidates"], api_key, timeout)
        except Exception as exc:
            print(f"imscore {prompt['id']} group failed: {exc!r}", flush=True)
            scores = {}
            for candidate in prompt["candidates"]:
                try:
                    single = call_imscore(prompt["prompt"], [candidate], api_key, timeout)
                    scores.update(single)
                except Exception as single_exc:
                    candidate["imscore_error"] = repr(single_exc)
                    print(f"imscore {prompt['id']} {candidate['id']} failed: {single_exc!r}", flush=True)
        for candidate in prompt["candidates"]:
            if candidate["id"] in scores:
                candidate["imscore_scores"] = scores[candidate["id"]]
                candidate.pop("imscore_error", None)
        done = sum(1 for candidate in prompt["candidates"] if candidate.get("imscore_scores"))
        print(prompt["id"], done, "/", len(prompt["candidates"]), flush=True)


def build_ood_prompts() -> list[dict[str, Any]]:
    if OOD_ASSET_DIR.exists():
        shutil.rmtree(OOD_ASSET_DIR)
    OOD_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    svg_prompt, svg_candidates = svg_complex_candidates()
    return [
        {
            "id": "ood-chart-infographic",
            "title": "Technical chart / infographic probe",
            "track": "ood",
            "ood_type": "Technical chart / infographic",
            "focus_dimension": "preference",
            "dimension_label": "Preference",
            "prompt": (
                "User Intent: Create a clear, polished technical chart or infographic that communicates data trends, "
                "key performance indicators, and structured business insights. Description: The image should use chart "
                "elements, labels, legends, and a clean presentation layout rather than a social poster or campaign graphic."
            ),
            "candidates": save_chart_assets(),
        },
        {
            "id": "ood-photorealistic",
            "title": "Photorealistic image probe",
            "track": "ood",
            "ood_type": "Photorealistic image",
            "focus_dimension": "preference",
            "dimension_label": "Preference",
            "prompt": (
                "User Intent: Evaluate a pure photorealistic image without typography, poster layout, or graphic design elements. "
                "Description: The image should look like a natural camera photograph with realistic lighting, depth, texture, and composition."
            ),
            "candidates": download_photo_assets(),
        },
        {
            "id": "ood-mobile-ui",
            "title": "Mobile app UI screen probe",
            "track": "ood",
            "ood_type": "Mobile app UI screen",
            "focus_dimension": "preference",
            "dimension_label": "Preference",
            "prompt": (
                "User Intent: Create a modern mobile app screen with clear hierarchy, navigation, cards, call-to-action, and polished product UI. "
                "Description: The image should resemble a real app screenshot or onboarding screen rather than a social media post."
            ),
            "candidates": save_mobile_assets(),
        },
        {
            "id": "ood-svg-complex",
            "title": "SVG complex vector probe",
            "track": "ood",
            "ood_type": "SVG complex vector illustration",
            "focus_dimension": "preference",
            "dimension_label": "Preference",
            "prompt": (
                f"User Intent: Recreate a complex vector illustration. Description: {svg_prompt}"
            ),
            "candidates": svg_candidates,
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=float(os.getenv("IMSCORE_TIMEOUT", "120")))
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    args = parser.parse_args()

    load_prod_env()
    api_key = args.api_key or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    ood_prompts = build_ood_prompts()
    score_taste_preference(ood_prompts)
    data["ood_prompts"] = ood_prompts
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    score_imscore(ood_prompts, api_key=api_key, timeout=args.timeout)
    data["ood_prompts"] = ood_prompts
    data["imscore_order"] = IMSCORE_ORDER
    data["imscore_labels"] = IMSCORE_LABELS
    data.setdefault("summary", {})["ood_prompts"] = len(ood_prompts)
    data["summary"]["ood_candidates"] = sum(len(prompt["candidates"]) for prompt in ood_prompts)
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ood_prompts": len(ood_prompts), "ood_candidates": data["summary"]["ood_candidates"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
