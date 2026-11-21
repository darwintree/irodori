#!/usr/bin/env python3
"""Export and prepare all Wave B event stories from datasets/selection_plan_sc29.json.

Handles stories shared across multiple characters cleanly by merging target speakers per story_id,
uses a single directory walk over assets/json, runs prepare.py's prepare_story in parallel,
and injects speaker-specific rules (e.g. fuyuko dual-persona triggers) into context.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/darwinwsl/code/irodori")
DEFAULT_PLAN = ROOT / "datasets/selection_plan_sc29.json"
DEFAULT_JSON_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/json")
DEFAULT_VOICE_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events")
DEFAULT_SOURCES_DIR = ROOT / "datasets/annotator_sources/events"
DEFAULT_WORK_DIR = ROOT / "datasets/annotator_work/events"
DEFAULT_CHARS_DIR = ROOT / "references/eidolon-sc/characters/facts"
DEFAULT_RULES_DIR = ROOT / "configs/speaker_caption_rules"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "voice-caption-annotator"))

from apply_speaker_rules import apply_to_work_dir
from export_sc_stories import convert_story
from prepare import prepare_story


def index_story_jsons(json_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for dirpath, _, filenames in os.walk(json_root):
        for fn in filenames:
            if fn.endswith(".json"):
                stem = fn[:-5]
                index[stem] = Path(dirpath) / fn
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Wave B SC event stories cleanly")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--json-root", type=Path, default=DEFAULT_JSON_ROOT)
    parser.add_argument("--voice-root", type=Path, default=DEFAULT_VOICE_ROOT)
    parser.add_argument("--sources-dir", type=Path, default=DEFAULT_SOURCES_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--chars-dir", type=Path, default=DEFAULT_CHARS_DIR)
    parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--overwrite-sources", action="store_true")
    parser.add_argument("--overwrite-work", action="store_true")
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    chars = plan["characters"]
    story_speakers: dict[str, set[str]] = {}
    for cid, info in chars.items():
        for sid in info.get("high_emo", []) + info.get("calm_daily", []):
            story_speakers.setdefault(str(sid), set()).add(str(cid))

    # Avoid clobbering pre-existing reviewed work packets (e.g. Mano's 400100103) when another
    # character in selection_plan_sc29 selects the same story_id: export only missing speakers
    # under packet_id "<story_id>__<missing_speakers>".
    packets: list[tuple[str, str, set[str]]] = []  # (packet_id, raw_story_id, target_speakers)
    for sid in sorted(story_speakers.keys()):
        target_sps = story_speakers[sid]
        existing_ann = args.work_dir / sid / "annotations.json"
        if existing_ann.is_file() and not args.overwrite_work:
            try:
                old = json.loads(existing_ann.read_text(encoding="utf-8"))
                covered = {str(it.get("speaker_id") or "") for it in old.get("items", [])}
            except Exception:
                covered = set()
            missing = target_sps - covered
            if not missing:
                packets.append((sid, sid, target_sps))
            else:
                pkt_id = f"{sid}__{'_'.join(sorted(missing))}"
                packets.append((pkt_id, sid, missing))
        else:
            packets.append((sid, sid, target_sps))

    print(f"[PLAN] {len(chars)} characters -> {len(packets)} story packets", flush=True)

    args.sources_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    need_export = [
        (pkt_id, raw_sid, sps)
        for pkt_id, raw_sid, sps in packets
        if args.overwrite_sources or not (args.sources_dir / f"{pkt_id}.json").is_file()
    ]
    if need_export:
        if not args.json_root.is_dir():
            print(f"[ERROR] JSON root not found: {args.json_root}", file=sys.stderr)
            return 2
        print(f"[INDEX] Walking {args.json_root} ...", flush=True)
        json_index = index_story_jsons(args.json_root)
        print(f"[INDEX] Found {len(json_index)} raw JSONs; exporting {len(need_export)} packets...", flush=True)

        exported = 0
        failed_export: list[str] = []

        def _export_one(pkt: tuple[str, str, set[str]]) -> tuple[str, bool, str]:
            pkt_id, raw_sid, sps = pkt
            jp = json_index.get(raw_sid)
            if jp is None:
                return pkt_id, False, "raw json missing"
            bundle = convert_story(jp, args.voice_root, sps, args.min_dur, args.max_dur)
            if bundle is None:
                return pkt_id, False, "no matching target clips"
            if pkt_id != raw_sid:
                bundle["story_id"] = pkt_id
                bundle["title"] = f"{bundle['meta'].get('category', 'event')}/{pkt_id}"
            out_p = args.sources_dir / f"{pkt_id}.json"
            out_p.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return pkt_id, True, f"{bundle['meta']['target_count']} clips"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_export_one, pkt) for pkt in need_export]
            for fut in as_completed(futs):
                pkt_id, ok, msg = fut.result()
                if ok:
                    exported += 1
                else:
                    failed_export.append(f"{pkt_id} ({msg})")
        print(f"[EXPORT] ok={exported}/{len(need_export)} failed={len(failed_export)}", flush=True)
        if failed_export:
            print(f"[EXPORT FAILURES] {failed_export}", file=sys.stderr)
            return 1
    else:
        print(f"[EXPORT] All {len(packets)} source JSONs already exist", flush=True)

    need_prepare = [
        pkt_id
        for pkt_id, _, _ in packets
        if args.overwrite_work or not (args.work_dir / pkt_id / "annotations.json").is_file()
    ]
    if need_prepare:
        print(f"[PREPARE] Preparing {len(need_prepare)} story work packets with {args.workers} workers...", flush=True)
        prepped = 0
        failed_prep: list[str] = []

        def _prep_one(pkt_id: str) -> tuple[str, bool, str]:
            try:
                prepare_story(
                    source_path=args.sources_dir / f"{pkt_id}.json",
                    work_root=args.work_dir,
                    characters_dir=args.chars_dir,
                    batch_size=12,
                    max_batch_sec=150.0,
                    silence_sec=0.8,
                    sample_rate=48000,
                    trim_silence=True,
                    overwrite=args.overwrite_work,
                )
                return pkt_id, True, "ok"
            except Exception as exc:
                return pkt_id, False, str(exc)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_prep_one, pkt_id) for pkt_id in need_prepare]
            for fut in as_completed(futs):
                pkt_id, ok, msg = fut.result()
                if ok:
                    prepped += 1
                    if prepped % 50 == 0 or prepped == len(need_prepare):
                        print(f"  prepared {prepped}/{len(need_prepare)}", flush=True)
                else:
                    failed_prep.append(f"{pkt_id} ({msg})")
        print(f"[PREPARE] ok={prepped}/{len(need_prepare)} failed={len(failed_prep)}", flush=True)
        if failed_prep:
            print(f"[PREPARE FAILURES] {failed_prep}", file=sys.stderr)
            return 1
    else:
        print(f"[PREPARE] All {len(packets)} story work packets already prepared", flush=True)

    n_rules = apply_to_work_dir(args.work_dir, args.rules_dir)
    print(f"[RULES] Newly patched context.md count: {n_rules}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
