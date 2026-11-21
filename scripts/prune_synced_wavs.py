#!/usr/bin/env python3
"""Safely prune intermediate WAV files from annotator work directories whose
annotations are already reviewed and verified inside datasets/catalog.sqlite.

Never deletes annotations.json, script.md, context.md, or run_*.log.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path("/home/darwinwsl/code/irodori")
sys.path.insert(0, str(ROOT / "scripts"))
from sync_catalog import derive_voice_key

DEFAULT_DB = ROOT / "datasets/catalog.sqlite"
ALLOWED_JSON_STAGES = {"reviewed", "human_done"}
ALLOWED_DB_STAGES = {"reviewed", "human_approved", "rejected"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune intermediate WAVs for synced, reviewed batches")
    parser.add_argument("--work-dir", action="append", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--exclude", action="append", default=[], help="Story/batch IDs to skip pruning")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    excluded = set(args.exclude)
    conn = sqlite3.connect(str(args.db))
    db_stages = dict(conn.execute("SELECT voice_key, caption_stage FROM clips").fetchall())
    conn.close()

    pruned_stories = 0
    pruned_files = 0
    freed_bytes = 0
    skipped_unverified = 0

    for wdir in args.work_dir:
        if not wdir.is_dir():
            continue
        for sdir in sorted(wdir.iterdir()):
            if not sdir.is_dir() or sdir.name in excluded:
                continue
            ann_path = sdir / "annotations.json"
            if not ann_path.is_file():
                continue
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if data.get("stage") not in ALLOWED_JSON_STAGES:
                continue

            items = data.get("items") or []
            if not items:
                continue
            all_synced = True
            for it in items:
                vkey = derive_voice_key(str(it.get("source_audio_path") or ""))
                if not vkey or db_stages.get(vkey) not in ALLOWED_DB_STAGES:
                    all_synced = False
                    break
            if not all_synced:
                skipped_unverified += 1
                continue

            wav_targets = list(sdir.glob("audio_batch_*.wav"))
            clips_dir = sdir / "clips"
            if clips_dir.is_dir():
                wav_targets.extend(clips_dir.glob("*.wav"))
            if not wav_targets:
                continue

            pruned_stories += 1
            for wp in wav_targets:
                try:
                    sz = wp.stat().st_size
                except OSError:
                    sz = 0
                freed_bytes += sz
                pruned_files += 1
                if not args.dry_run:
                    wp.unlink(missing_ok=True)
            if not args.dry_run and clips_dir.is_dir():
                try:
                    clips_dir.rmdir()
                except OSError:
                    pass

    mode = "DRY-RUN" if args.dry_run else "PRUNED"
    print(
        f"[{mode}] stories={pruned_stories} files={pruned_files} "
        f"freed={freed_bytes / (1024**3):.2f} GiB | skipped_unsynced={skipped_unverified}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
