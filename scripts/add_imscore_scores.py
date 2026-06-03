from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
import yaml


PROJECT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT / "data.json"
PROD_ENV_PATH = Path("/home/ubuntu/web-app/ml-engine/app/prod-env.yaml")
IMSCORE_ENDPOINT = "https://model-wnpgy843.api.baseten.co/environments/production/predict"

IMSCORE_ORDER = ["hpsv21", "pickscore", "clipscore", "imagereward", "laion_aesthetic"]
IMSCORE_LABELS = {
    "hpsv21": "HPS v2.1",
    "pickscore": "PickScore",
    "clipscore": "CLIPScore",
    "imagereward": "ImageReward",
    "laion_aesthetic": "LAION aesthetic",
}


def load_prod_env(path: Path = PROD_ENV_PATH) -> None:
    if not path.exists():
        return
    data = yaml.safe_load(path.read_text()) or {}
    for key, value in (data.get("env_variables") or {}).items():
        if value is not None:
            os.environ.setdefault(key, str(value))


def image_data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def call_imscore(prompt: str, candidates: list[dict[str, Any]], api_key: str, timeout: float) -> dict[str, dict[str, float]]:
    response = requests.post(
        IMSCORE_ENDPOINT,
        headers={"Authorization": f"Api-Key {api_key}"},
        json={
            "prompt": prompt,
            "metrics": IMSCORE_ORDER,
            "candidates": [
                {
                    "id": candidate["id"],
                    "image": image_data_uri(PROJECT / candidate["asset"]),
                }
                for candidate in candidates
            ],
        },
        timeout=timeout,
    )
    if not response.ok:
        raise requests.HTTPError(response.text[:500], response=response)
    raw = response.json().get("scores", {})
    return {
        candidate_id: {metric: float(value) for metric, value in scores.items()}
        for candidate_id, scores in raw.items()
    }


def score_prompt(prompt: dict[str, Any], api_key: str, timeout: float, force: bool) -> None:
    candidates = prompt.get("candidates", [])
    missing = [
        candidate
        for candidate in candidates
        if force or any(candidate.get("imscore_scores", {}).get(metric) is None for metric in IMSCORE_ORDER)
    ]
    if not missing:
        return

    try:
        scores = call_imscore(prompt["prompt"], missing, api_key, timeout)
    except Exception as exc:
        if len(missing) == 1:
            missing[0]["imscore_error"] = repr(exc)
            return
        for candidate in missing:
            try:
                scores = call_imscore(prompt["prompt"], [candidate], api_key, timeout)
                candidate["imscore_scores"] = scores.get(candidate["id"], {})
                candidate.pop("imscore_error", None)
            except Exception as single_exc:
                candidate["imscore_error"] = repr(single_exc)
        return

    for candidate in missing:
        candidate["imscore_scores"] = scores.get(candidate["id"], {})
        candidate.pop("imscore_error", None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=os.getenv("BASETEN_API_KEY", ""))
    parser.add_argument("--timeout", type=float, default=float(os.getenv("IMSCORE_TIMEOUT", "300")))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    load_prod_env()
    api_key = args.api_key or os.environ.get("BASETEN_API_KEY")
    if not api_key:
        raise RuntimeError("Set BASETEN_API_KEY or pass --api-key.")

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    data["imscore_order"] = IMSCORE_ORDER
    data["imscore_labels"] = IMSCORE_LABELS
    for prompt in data.get("prompts", []):
        score_prompt(prompt, api_key=api_key, timeout=args.timeout, force=args.force)
        data.setdefault("summary", {})["imscore_scored"] = all(
            all(candidate.get("imscore_scores", {}).get(metric) is not None for metric in IMSCORE_ORDER)
            for row in data.get("prompts", [])
            for candidate in row.get("candidates", [])
        )
        DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        done = sum(1 for c in prompt.get("candidates", []) if c.get("imscore_scores"))
        print(prompt["id"], done, "/", len(prompt.get("candidates", [])), flush=True)

    data.setdefault("summary", {})["imscore_scored"] = all(
        all(candidate.get("imscore_scores", {}).get(metric) is not None for metric in IMSCORE_ORDER)
        for row in data.get("prompts", [])
        for candidate in row.get("candidates", [])
    )
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
