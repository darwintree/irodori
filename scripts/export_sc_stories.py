#!/usr/bin/env python3
"""Convert Shiny Colors raw story JSONs into generic voice-caption-annotator source JSONs.

Lives in the parent workspace (outside the open-source voice-caption-annotator submodule).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_JSON_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/json")
DEFAULT_VOICE_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events")

SPEAKER_TO_ID = {
    "真乃": "mano",
    "灯織": "hiori",
    "めぐる": "meguru",
    "恋鐘": "kogane",
    "摩美々": "mamimi",
    "咲耶": "sakuya",
    "結華": "yuika",
    "霧子": "kiriko",
    "果穂": "kaho",
    "智代子": "chiyoko",
    "樹里": "juri",
    "凛世": "rinze",
    "夏葉": "natsuha",
    "甘奈": "amana",
    "甜花": "tenka",
    "千雪": "chiyuki",
    "あさひ": "asahi",
    "冬優子": "fuyuko",
    "愛依": "mei",
    "透": "toru",
    "円香": "madoka",
    "小糸": "koito",
    "雛菜": "hinana",
    "にちか": "nichika",
    "美琴": "mikoto",
    "ルカ": "luca",
    "羽那": "hana",
    "はるき": "haruki",
    "はづき": "hazuki",
}


def probe_duration(audio_path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return None
    try:
        return float(res.stdout.strip())
    except ValueError:
        return None


def clean_cue(val: Any) -> str | None:
    if not isinstance(val, str):
        return None
    v = val.strip()
    if not v or v.lower() == "null" or v.startswith("wait"):
        return None
    return v


def convert_story(
    json_path: Path,
    voice_root: Path,
    target_speakers: set[str] | None,
    min_dur: float,
    max_dur: float,
) -> dict[str, Any] | None:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return None

    story_id = json_path.stem
    category = json_path.parent.name
    lines: list[dict[str, Any]] = []
    target_count = 0

    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            continue

        speaker = str(entry.get("speaker") or "ナレーション").strip()
        char_label = entry.get("charLabel")
        if speaker in SPEAKER_TO_ID:
            speaker_id = SPEAKER_TO_ID[speaker]
        elif speaker in {"プロデューサー", "ナレーション"}:
            speaker_id = None
        elif isinstance(char_label, str) and char_label.strip():
            speaker_id = char_label.strip().lower()
        else:
            speaker_id = None

        cues: dict[str, str] = {}
        for src_k, dst_k in (("charAnim1", "anim1"), ("charAnim2", "expression"), ("charLipAnim", "lip")):
            v = clean_cue(entry.get(src_k))
            if v:
                cues[dst_k] = v

        voice_rel = entry.get("voice")
        audio_path_str: str | None = None
        dur: float | None = None
        annotate = False

        if isinstance(voice_rel, str) and voice_rel.strip():
            cand = voice_root / f"{voice_rel.strip()}.m4a"
            if cand.is_file():
                audio_path_str = str(cand)
                speaker_ok = (
                    not target_speakers
                    or speaker in target_speakers
                    or (speaker_id is not None and speaker_id in target_speakers)
                )
                if speaker_ok:
                    dur = probe_duration(cand)
                    if dur is not None and min_dur <= dur <= max_dur:
                        annotate = True
                        target_count += 1

        line_obj: dict[str, Any] = {
            "line_index": len(lines),
            "speaker": speaker,
            "text": text.replace("\r\n", "\n").strip(),
            "audio_path": audio_path_str,
            "annotate": annotate,
            "cues": cues,
        }
        if speaker_id:
            line_obj["speaker_id"] = speaker_id
        if entry.get("id") is not None:
            line_obj["id"] = str(entry["id"])
        if dur is not None:
            line_obj["duration"] = round(dur, 3)

        lines.append(line_obj)

    if target_count == 0:
        return None

    return {
        "story_id": story_id,
        "title": f"{category}/{story_id}",
        "meta": {
            "source": "shinycolors",
            "category": category,
            "raw_json": str(json_path),
            "target_count": target_count,
        },
        "lines": lines,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Shiny Colors raw JSONs to generic annotator source format.")
    parser.add_argument("--json-file", action="append", type=Path, default=[], help="Specific raw JSON file(s)")
    parser.add_argument("--story-id", action="append", default=[], help="Story ID(s) to search under --json-root")
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT, help="SC raw JSON root directory")
    parser.add_argument("--voice-root", type=Path, default=DEFAULT_VOICE_ROOT, help="SC voice events root directory")
    parser.add_argument("--target-speaker", action="append", default=[], help="Filter target speaker(s) (e.g. 真乃 or mano)")
    parser.add_argument("--min-dur", type=float, default=3.0, help="Min target audio duration (seconds)")
    parser.add_argument("--max-dur", type=float, default=15.0, help="Max target audio duration (seconds)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for generic source JSONs")
    args = parser.parse_args()

    files: list[Path] = list(args.json_file)
    for sid in args.story_id:
        matches = sorted(args.json_root.rglob(f"{sid}.json"))
        if not matches:
            print(f"[WARN] story_id {sid} not found under {args.json_root}", file=sys.stderr)
        files.extend(matches)

    if not files:
        parser.error("Provide --json-file or --story-id")

    target_speakers = set(args.target_speaker) if args.target_speaker else None
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported = 0

    for fp in files:
        bundle = convert_story(fp, args.voice_root, target_speakers, args.min_dur, args.max_dur)
        if bundle is None:
            print(f"[SKIP] {fp.name}: no matching target clips")
            continue
        out_path = args.out_dir / f"{bundle['story_id']}.json"
        out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] {bundle['story_id']} -> {out_path} ({bundle['meta']['target_count']} target clips)")
        exported += 1

    print(f"Exported {exported}/{len(files)} story source file(s).")
    return 0 if exported > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
