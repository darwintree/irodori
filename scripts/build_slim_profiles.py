#!/usr/bin/env python3
"""Extract concise voice-acting quick-reference profiles for all 29 SC speakers."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEAKERS = [
    "mano", "hiori", "meguru", "kogane", "mamimi", "sakuya", "yuika", "kiriko",
    "kaho", "chiyoko", "juri", "rinze", "natsuha", "amana", "tenka", "chiyuki",
    "asahi", "fuyuko", "mei", "toru", "madoka", "koito", "hinana", "nichika",
    "mikoto", "luca", "hana", "haruki", "hazuki",
]
KEEP_SECTIONS = ("身份", "自称", "对他人的称呼", "口癖")


def extract_slim_profile(text: str) -> str:
    parts = re.split(r"(?m)^## ", text)
    kept = [parts[0].strip()]
    for sec in parts[1:]:
        heading = sec.splitlines()[0].strip()
        if any(k in heading for k in KEEP_SECTIONS):
            kept.append("## " + sec.strip())
    return "\n\n".join(kept).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build slim SC29 voice profiles from character sheets.")
    parser.add_argument(
        "--sheets-dir",
        type=Path,
        default=Path("/home/darwinwsl/code/irodori/references/eidolon-sc/characters/sheets"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/darwinwsl/code/irodori/references/sc29_voice_profiles"),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sizes: list[tuple[str, int]] = []

    for spk in SPEAKERS:
        src = args.sheets_dir / f"{spk}.md"
        if not src.is_file():
            print(f"[ERROR] Missing sheet: {src}", file=sys.stderr)
            return 1
        slim = extract_slim_profile(src.read_text(encoding="utf-8"))
        dst = args.out_dir / f"{spk}.md"
        dst.write_text(slim, encoding="utf-8")
        sizes.append((spk, len(slim.encode("utf-8"))))

    min_s = min(sz for _, sz in sizes) / 1024.0
    avg_s = sum(sz for _, sz in sizes) / len(sizes) / 1024.0
    max_s = max(sz for _, sz in sizes) / 1024.0
    print(f"[SLIM-PROFILES] built={len(sizes)}/{len(SPEAKERS)} in {args.out_dir} | min={min_s:.1f}KB avg={avg_s:.1f}KB max={max_s:.1f}KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
