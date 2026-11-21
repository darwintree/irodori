#!/usr/bin/env python3
"""Export Shiny Colors P0+P1 character voices (assets/sounds/voice/characters/<001..028>)
into generic voice-caption-annotator source JSON batches (asr_mode=True).

Priority P0+P1 includes:
- P0: casting_voice, mypage_talking_multi (duo/trio/quartet/quintet), explicit emotion tags
- P1: minigame (daifugo/283stage), reaction/touch, live/audition/skill, reliance, idol_rank_s_to_f
Excludes: birthday, prelude_voice, unison, and durations outside [min_dur, max_dur].
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_CHAR_VOICE_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/characters")

IDOLS: list[tuple[str, str, str]] = [
    ("001", "真乃", "mano"), ("002", "灯織", "hiori"), ("003", "めぐる", "meguru"),
    ("004", "恋鐘", "kogane"), ("005", "摩美々", "mamimi"), ("006", "咲耶", "sakuya"),
    ("007", "結華", "yuika"), ("008", "霧子", "kiriko"),
    ("009", "果穂", "kaho"), ("010", "智代子", "chiyoko"), ("011", "樹里", "juri"),
    ("012", "凛世", "rinze"), ("013", "夏葉", "natsuha"),
    ("014", "甘奈", "amana"), ("015", "甜花", "tenka"), ("016", "千雪", "chiyuki"),
    ("017", "あさひ", "asahi"), ("018", "冬優子", "fuyuko"), ("019", "愛依", "mei"),
    ("020", "透", "toru"), ("021", "円香", "madoka"), ("022", "小糸", "koito"),
    ("023", "雛菜", "hinana"),
    ("024", "にちか", "nichika"), ("025", "美琴", "mikoto"),
    ("026", "ルカ", "luca"), ("027", "羽那", "hana"), ("028", "はるき", "haruki"),
]

CODE_TO_IDOL = {c: (ja, sid) for c, ja, sid in IDOLS}
SID_TO_CODE = {sid: c for c, _, sid in IDOLS}
JA_TO_CODE = {ja: c for c, ja, _ in IDOLS}


def parse_mp4_duration(path: Path) -> float | None:
    try:
        buf = path.read_bytes()
    except OSError:
        return None
    idx = buf.find(b"mvhd")
    if idx == -1 or idx + 28 > len(buf):
        return None
    version = buf[idx + 4]
    if version == 0:
        timescale, duration = struct.unpack(">II", buf[idx + 16 : idx + 24])
        return duration / timescale if timescale > 0 else None
    if version == 1 and idx + 36 <= len(buf):
        timescale = struct.unpack(">I", buf[idx + 24 : idx + 28])[0]
        duration = struct.unpack(">Q", buf[idx + 28 : idx + 36])[0]
        return duration / timescale if timescale > 0 else None
    return None


def classify_voice(filename: str) -> tuple[str, str] | None:
    """Return (priority, bucket_key) for P0+P1, or None if excluded/lower priority."""
    if re.search(r"(unison|birthday|prelude_voice)", filename, re.I):
        return None
    if re.search(r"casting_voice", filename, re.I):
        return ("P0", "casting")
    if re.search(
        r"(anger|sad|shy|cry|sigh|laugh|surp|happy|smile|trouble|sorry|panic|fear|mad|joy|excited|agree)",
        filename,
        re.I,
    ):
        return ("P0", "emotion_tag")
    if re.search(r"mypage_talking_(duo|trio|quartet|quintet)", filename, re.I):
        return ("P0", "talking_multi")
    if re.search(r"reliance", filename, re.I):
        return ("P1", "reliance")
    if re.search(r"(reaction|_(head|belly|breast)_)", filename, re.I):
        return ("P1", "reaction")
    if re.search(r"(daifugo_|283stage_)", filename, re.I):
        return ("P1", "minigame")
    if re.search(r"idol_rank_[a-fs]", filename, re.I):
        return ("P1", "idol_rank")
    if re.search(r"(audition|concert|live|encore|appeal|skill|link|memory|fes|judge)", filename, re.I):
        return ("P1", "live_skill")
    return None


def resolve_codes(targets: list[str]) -> list[str]:
    if not targets:
        return [c for c, _, _ in IDOLS]
    out: list[str] = []
    for t in targets:
        if t in CODE_TO_IDOL:
            out.append(t)
        elif t in SID_TO_CODE:
            out.append(SID_TO_CODE[t])
        elif t in JA_TO_CODE:
            out.append(JA_TO_CODE[t])
        else:
            raise ValueError(f"Unknown character target: {t}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export P0+P1 SC character voices into ASR+caption source batches.")
    parser.add_argument("--char-root", type=Path, default=DEFAULT_CHAR_VOICE_ROOT)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--character", action="append", default=[], help="Character code/id/ja (repeatable; default all 28)")
    parser.add_argument("--bucket", action="append", default=[], help="Optional filter bucket(s), e.g. casting, talking_multi")
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=15.0)
    parser.add_argument("--batch-size", type=int, default=20, help="Max clips per batch JSON")
    args = parser.parse_args()

    codes = resolve_codes(args.character)
    allowed_buckets = set(args.bucket) if args.bucket else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    total_batches = 0
    total_clips = 0

    for code in codes:
        ja, sid = CODE_TO_IDOL[code]
        cdir = args.char_root / code
        if not cdir.is_dir():
            print(f"[WARN] missing dir {cdir}", file=sys.stderr)
            continue

        grouped: dict[str, list[tuple[str, Path, float, str]]] = defaultdict(list)
        for fp in sorted(cdir.glob("*.m4a")):
            cls = classify_voice(fp.name)
            if cls is None:
                continue
            prio, bucket = cls
            if allowed_buckets and bucket not in allowed_buckets:
                continue
            dur = parse_mp4_duration(fp)
            if dur is None or dur < args.min_dur or dur > args.max_dur:
                continue
            grouped[bucket].append((fp.name, fp, dur, prio))

        for bucket, entries in sorted(grouped.items()):
            for chunk_idx in range(0, len(entries), args.batch_size):
                chunk = entries[chunk_idx : chunk_idx + args.batch_size]
                part = chunk_idx // args.batch_size + 1
                suffix = f"_{part:02d}" if len(entries) > args.batch_size else ""
                story_id = f"char_{code}_{sid}_{bucket}{suffix}"
                lines: list[dict[str, Any]] = []
                for li, (fname, fp, dur, prio) in enumerate(chunk):
                    lines.append({
                        "line_index": li,
                        "id": Path(fname).stem,
                        "speaker": ja,
                        "speaker_id": sid,
                        "text": "",
                        "audio_path": str(fp),
                        "duration": round(dur, 3),
                        "annotate": True,
                        "cues": {
                            "priority": prio,
                            "voice_category": bucket,
                            "filename": fname,
                        },
                    })

                bundle = {
                    "story_id": story_id,
                    "title": f"characters/{code}_{sid}/{bucket}{suffix}",
                    "meta": {
                        "source": "shinycolors_characters",
                        "asr_mode": True,
                        "character_code": code,
                        "speaker": ja,
                        "speaker_id": sid,
                        "voice_category": bucket,
                        "target_count": len(lines),
                    },
                    "lines": lines,
                }
                out_path = args.out_dir / f"{story_id}.json"
                out_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                total_batches += 1
                total_clips += len(lines)

    print(f"Exported {total_batches} batch(es), {total_clips} clip(s) -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
