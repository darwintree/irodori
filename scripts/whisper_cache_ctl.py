#!/usr/bin/env python3
"""Stash and restore pre-computed whisper_text fields for prepared annotator batches.

Keeps Pass 1 LLM transcription 100% blind to local Whisper output while allowing
Whisper large-v3 to be pre-computed on an otherwise idle GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SKIP_RESTORE_STAGES = {"reviewed", "human_done"}


def _atomic_write_json(path: Path, data: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def cmd_stash(args: argparse.Namespace) -> int:
    work_dir: Path = args.work_dir.resolve()
    cache_path: Path = (args.cache or (work_dir / ".whisper_cache_p1.json")).resolve()
    if not work_dir.is_dir():
        print(f"Work directory does not exist: {work_dir}", file=sys.stderr)
        return 1

    prepared_docs: list[tuple[str, Path, dict]] = []
    non_empty_whisper = 0
    total_items = 0

    for entry in sorted(work_dir.iterdir()):
        ann_path = entry / "annotations.json"
        if not entry.is_dir() or not ann_path.is_file():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if data.get("stage") != "prepared":
            continue
        prepared_docs.append((entry.name, ann_path, data))
        for it in data.get("items", []):
            total_items += 1
            if str(it.get("whisper_text") or "").strip():
                non_empty_whisper += 1

    if non_empty_whisper == 0 and not args.allow_empty:
        print(
            f"Aborting stash: 0 non-empty whisper_text across {len(prepared_docs)} prepared story(s) "
            f"({total_items} items). Pass --allow-empty to override.",
            file=sys.stderr,
        )
        return 1

    merged_cache: dict[str, dict[str, str]] = {}
    if cache_path.is_file():
        existing = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            for sid, imap in existing.items():
                if isinstance(imap, dict):
                    merged_cache[str(sid)] = {str(k): str(v or "") for k, v in imap.items()}

    for sid, _, data in prepared_docs:
        story_cache = merged_cache.setdefault(sid, {})
        for it in data.get("items", []):
            iid = str(it.get("id"))
            wtxt = str(it.get("whisper_text") or "")
            if wtxt.strip() or iid not in story_cache:
                story_cache[iid] = wtxt

    _atomic_write_json(cache_path, merged_cache)

    cleared_items = 0
    for _, ann_path, data in prepared_docs:
        changed = False
        for it in data.get("items", []):
            if it.get("whisper_text") != "":
                it["whisper_text"] = ""
                changed = True
            cleared_items += 1
        if changed:
            _atomic_write_json(ann_path, data)

    print(
        f"stashed_stories={len(prepared_docs)} stashed_items={cleared_items} "
        f"non_empty_whisper={non_empty_whisper} cache_path={cache_path}",
        flush=True,
    )
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    work_dir: Path = args.work_dir.resolve()
    cache_path: Path = (args.cache or (work_dir / ".whisper_cache_p1.json")).resolve()
    if not cache_path.is_file():
        print(f"Cache file not found: {cache_path}", file=sys.stderr)
        return 1

    cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(cache_data, dict):
        print(f"Invalid cache JSON structure in {cache_path}", file=sys.stderr)
        return 1

    restored_stories = 0
    restored_items = 0
    non_empty_whisper = 0
    skipped_reviewed = 0
    missing_items = 0

    for sid, item_map in sorted(cache_data.items()):
        if not isinstance(item_map, dict):
            continue
        ann_path = work_dir / sid / "annotations.json"
        if not ann_path.is_file():
            missing_items += len(item_map)
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if data.get("stage") in SKIP_RESTORE_STAGES:
            skipped_reviewed += 1
            continue

        changed = False
        for it in data.get("items", []):
            iid = str(it.get("id"))
            if iid not in item_map:
                missing_items += 1
                continue
            cur = str(it.get("whisper_text") or "").strip()
            cached = str(item_map[iid] or "")
            if args.force or not cur:
                if it.get("whisper_text") != cached:
                    it["whisper_text"] = cached
                    changed = True
                restored_items += 1
                if cached.strip():
                    non_empty_whisper += 1

        if changed:
            _atomic_write_json(ann_path, data)
            restored_stories += 1

    print(
        f"restored_stories={restored_stories} restored_items={restored_items} "
        f"non_empty_whisper={non_empty_whisper} skipped_reviewed={skipped_reviewed} "
        f"missing_items={missing_items}",
        flush=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stash and restore cached whisper_text for prepared batches.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_stash = sub.add_parser("stash", help="Save whisper_text for stage==prepared stories and clear JSON fields")
    p_stash.add_argument("--work-dir", required=True, type=Path)
    p_stash.add_argument("--cache", default=None, type=Path)
    p_stash.add_argument("--allow-empty", action="store_true")

    p_restore = sub.add_parser("restore", help="Restore cached whisper_text into non-reviewed stories")
    p_restore.add_argument("--work-dir", required=True, type=Path)
    p_restore.add_argument("--cache", default=None, type=Path)
    p_restore.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.subcommand == "stash":
        return cmd_stash(args)
    return cmd_restore(args)


if __name__ == "__main__":
    sys.exit(main())
