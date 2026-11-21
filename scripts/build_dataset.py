#!/usr/bin/env python3
"""Build versioned Irodori-TTS train/valid manifests from datasets/catalog.sqlite using a YAML recipe.

Supports --plan-only to inspect exact per-speaker counts, min:max balance ratio, and caption coverage
before wav/latent extraction.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_recipe(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train/valid manifests from catalog.sqlite")
    parser.add_argument("--recipe", type=Path, required=True, help="Path to dataset YAML recipe")
    parser.add_argument("--plan-only", action="store_true", help="Compute stats & balance report without requiring latent_path")
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    db_path = Path(recipe["catalog_db"])
    out_dir = Path(recipe["output_dir"])
    speakers: list[str] = recipe["speakers"]
    min_dur = float(recipe.get("min_duration", 2.0))
    max_dur = float(recipe.get("max_duration", 15.0))
    dedup_by_text = bool(recipe.get("dedup_by_text", True))
    max_per_speaker = int(recipe.get("max_per_speaker", 6480))
    speaker_oversample: dict[str, int] = recipe.get("speaker_oversample") or {}
    caption_oversample = int(recipe.get("caption_oversample", 3))
    valid_stages = set(recipe.get("include_caption_stages") or ["reviewed", "human_approved"])
    valid_ratio = float(recipe.get("valid_ratio", 0.02))
    min_valid = int(recipe.get("min_valid_per_speaker", 15))
    seed = int(recipe.get("seed", 42))

    if not db_path.is_file():
        print(f"[ERROR] Catalog DB not found: {db_path}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in speakers)
    rows = conn.execute(
        f"""
        SELECT voice_key, speaker_id, source_type, category, duration,
               text_final, text_hash, emo_score, caption_ja, caption_stage,
               raw_m4a_path, wav_path, latent_path
        FROM clips
        WHERE speaker_id IN ({placeholders})
          AND duration >= ? AND duration <= ?
          AND COALESCE(caption_stage, 'none') != 'rejected'
          AND (text_final IS NOT NULL AND TRIM(text_final) != '')
        """,
        (*speakers, min_dur, max_dur),
    ).fetchall()
    conn.close()

    by_sp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        d = dict(r)
        if not args.plan_only and not d.get("latent_path"):
            continue
        has_cap = (d.get("caption_stage") in valid_stages) and bool((d.get("caption_ja") or "").strip())
        d["has_cap"] = has_cap
        by_sp[d["speaker_id"]].append(d)

    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    selected_keys: list[str] = []
    per_speaker_stats: list[dict[str, Any]] = []

    for sp in speakers:
        pool = by_sp.get(sp, [])
        # Deduplicate by text_hash: prefer captioned > higher emo_score > longer duration
        if dedup_by_text:
            best_by_hash: dict[str, dict[str, Any]] = {}
            for item in pool:
                h = item["text_hash"] or item["voice_key"]
                prev = best_by_hash.get(h)
                key_cur = (1 if item["has_cap"] else 0, item["emo_score"] or 0, item["duration"] or 0.0)
                if prev is None:
                    best_by_hash[h] = item
                else:
                    key_prev = (1 if prev["has_cap"] else 0, prev["emo_score"] or 0, prev["duration"] or 0.0)
                    if key_cur > key_prev:
                        best_by_hash[h] = item
            pool = list(best_by_hash.values())

        # Priority selection up to max_per_speaker:
        # 1) All captioned clips, 2) High-emotion clips, 3) Stratified random sample of remaining
        captioned = [x for x in pool if x["has_cap"]]
        uncap_emo = [x for x in pool if not x["has_cap"] and (x["emo_score"] or 0) > 0]
        uncap_plain = [x for x in pool if not x["has_cap"] and (x["emo_score"] or 0) <= 0]
        rng.shuffle(captioned)
        rng.shuffle(uncap_emo)
        rng.shuffle(uncap_plain)

        ordered = captioned + uncap_emo + uncap_plain
        kept = ordered[:max_per_speaker]
        for x in kept:
            selected_keys.append(x["voice_key"])

        # Split valid (stratified: never oversample valid)
        n_val = min(len(kept) // 10, max(min_valid, int(round(len(kept) * valid_ratio))))
        rng.shuffle(kept)
        val_set = kept[:n_val]
        trn_set = kept[n_val:]

        sp_mult = int(speaker_oversample.get(sp, 1))
        sp_train_rows = 0
        sp_cap_train_rows = 0
        for x in trn_set:
            mult = sp_mult * (caption_oversample if x["has_cap"] else 1)
            row = {
                "voice_key": x["voice_key"],
                "latent_path": x["latent_path"],
                "text": x["text_final"],
                "speaker_id": sp,
                "caption": x["caption_ja"] if x["has_cap"] else "",
            }
            for _ in range(mult):
                train_rows.append(row)
                sp_train_rows += 1
                if x["has_cap"]:
                    sp_cap_train_rows += 1

        for x in val_set:
            valid_rows.append({
                "voice_key": x["voice_key"],
                "latent_path": x["latent_path"],
                "text": x["text_final"],
                "speaker_id": sp,
                "caption": x["caption_ja"] if x["has_cap"] else "",
            })

        per_speaker_stats.append({
            "speaker_id": sp,
            "dedup_pool": len(pool),
            "kept_unique": len(kept),
            "kept_captioned": sum(1 for x in kept if x["has_cap"]),
            "train_rows": sp_train_rows,
            "train_cap_rows": sp_cap_train_rows,
            "valid_rows": len(val_set),
            "sp_oversample": sp_mult,
        })

    non_zero = [s for s in per_speaker_stats if s["kept_unique"] > 0]
    eff_counts = [s["kept_unique"] * s["sp_oversample"] for s in non_zero]
    min_eff = min(eff_counts) if eff_counts else 0
    max_eff = max(eff_counts) if eff_counts else 0
    ratio = round(min_eff / max_eff, 4) if max_eff else 0.0
    total_cap_rows = sum(s["train_cap_rows"] for s in per_speaker_stats)

    summary = {
        "recipe": recipe["name"],
        "plan_only": args.plan_only,
        "num_speakers": len(non_zero),
        "total_kept_unique": sum(s["kept_unique"] for s in per_speaker_stats),
        "total_kept_captioned": sum(s["kept_captioned"] for s in per_speaker_stats),
        "total_train_rows": len(train_rows),
        "total_valid_rows": len(valid_rows),
        "effective_caption_ratio": round(total_cap_rows / len(train_rows), 4) if train_rows else 0.0,
        "min_effective_speaker": min_eff,
        "max_effective_speaker": max_eff,
        "min_max_ratio": ratio,
        "per_speaker": per_speaker_stats,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "selected_voice_keys.txt").write_text("\n".join(selected_keys) + "\n", encoding="utf-8")

    if not args.plan_only:
        rng.shuffle(train_rows)
        with (out_dir / "train_manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in train_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with (out_dir / "valid_manifest.jsonl").open("w", encoding="utf-8") as f:
            for r in valid_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[{recipe['name']}] speakers={summary['num_speakers']} | unique={summary['total_kept_unique']} "
        f"| captioned={summary['total_kept_captioned']} | train_rows={summary['total_train_rows']} "
        f"(cap_ratio={summary['effective_caption_ratio']*100:.1f}%) | min/max={min_eff}/{max_eff} ({ratio:.3f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
