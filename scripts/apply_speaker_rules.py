#!/usr/bin/env python3
"""Inject project-side speaker caption rules (configs/speaker_caption_rules/<sid>.md)
into prepared work/<batch_id>/context.md files without modifying generic voice-caption-annotator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_RULES_DIR = Path("/home/darwinwsl/code/irodori/configs/speaker_caption_rules")
MARKER = "<!-- SPEAKER_CAPTION_RULES -->"


def apply_to_work_dir(work_dir: Path, rules_dir: Path) -> int:
    rules: dict[str, str] = {}
    if rules_dir.is_dir():
        for p in sorted(rules_dir.glob("*.md")):
            rules[p.stem.lower()] = p.read_text(encoding="utf-8").strip()
    if not rules:
        return 0

    patched = 0
    for ann_path in sorted(work_dir.glob("*/annotations.json")):
        ctx_path = ann_path.parent / "context.md"
        if not ctx_path.is_file():
            continue
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        sids = {str(it.get("speaker_id") or "").lower() for it in data.get("items", [])}
        matched = [rules[s] for s in sorted(sids) if s in rules]
        if not matched:
            continue
        cur = ctx_path.read_text(encoding="utf-8")
        if MARKER in cur:
            continue
        inject_block = f"\n{MARKER}\n" + "\n\n".join(matched) + "\n\n---\n"
        # Insert right after the first heading block so Pass 1 & Pass 2 see it immediately
        lines = cur.splitlines(keepends=True)
        insert_at = 0
        for idx, line in enumerate(lines):
            if line.startswith("## "):
                insert_at = idx
                break
        else:
            insert_at = len(lines)
        new_text = "".join(lines[:insert_at]) + inject_block + "".join(lines[insert_at:])
        ctx_path.write_text(new_text, encoding="utf-8")
        patched += 1
    return patched


def main() -> int:
    parser = argparse.ArgumentParser(description="Inject speaker-specific caption rules into work/*/context.md")
    parser.add_argument("--work-dir", action="append", type=Path, required=True)
    parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    args = parser.parse_args()

    total = 0
    for wd in args.work_dir:
        n = apply_to_work_dir(wd, args.rules_dir)
        print(f"[RULES] {wd}: patched {n} context.md file(s)")
        total += n
    print(f"Total patched: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
