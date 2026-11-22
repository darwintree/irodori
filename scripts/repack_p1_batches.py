#!/usr/bin/env python3
"""Repack fragmented `prepared` character batches into denser <=25-clip / <=180s batches
with slim voice profiles and isolated `.whisper_cache_p1.json` (zero ffmpeg re-encoding)."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/darwinwsl/code/irodori")
sys.path.insert(0, str(ROOT / "voice-caption-annotator"))
from prepare import concat_batch_wav, fmt_ts, safe_filename  # noqa: E402


def _load_whisper_lookup(cache_path: Path | None) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    by_story: dict[str, dict[str, str]] = {}
    by_item: dict[str, str] = {}
    if not cache_path:
        return by_story, by_item
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, dict):
                smap = {str(iid): str(wt or "") for iid, wt in v.items()}
                by_story[str(k)] = smap
                for iid, wt in smap.items():
                    if wt.strip() or iid not in by_item:
                        by_item[iid] = wt
            elif isinstance(v, str):
                by_item[str(k)] = v
    return by_story, by_item


def _pack_speaker_batches(
    speaker_items: list[dict[str, Any]],
    max_items: int,
    max_sec: float,
    max_batches: int | None,
) -> list[list[dict[str, Any]]]:
    packed: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_sec = 0.0
    for entry in speaker_items:
        dur = float(entry["item"].get("duration") or 0.0)
        if cur and (len(cur) >= max_items or cur_sec + dur > max_sec):
            packed.append(cur)
            if max_batches is not None and len(packed) >= max_batches:
                return packed
            cur = []
            cur_sec = 0.0
        cur.append(entry)
        cur_sec += dur
    if cur and (max_batches is None or len(packed) < max_batches):
        packed.append(cur)
    return packed


def main() -> int:
    parser = argparse.ArgumentParser(description="Repack prepared character batches with slim profiles.")
    parser.add_argument("--src-dir", type=Path, default=ROOT / "datasets/annotator_work/characters")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--profiles-dir", type=Path, default=ROOT / "references/sc29_voice_profiles")
    parser.add_argument("--whisper-cache", type=Path, default=None)
    parser.add_argument("--speakers", default=None, help="Comma-separated speaker IDs to include")
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--max-sec", type=float, default=120.0)
    parser.add_argument("--silence-sec", type=float, default=0.8)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--max-batches-per-speaker", type=int, default=None)
    parser.add_argument("--require-whisper-ratio", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    speaker_filter = {s.strip() for s in args.speakers.split(",") if s.strip()} if args.speakers else None
    cache_by_story, cache_by_item = _load_whisper_lookup(args.whisper_cache)

    by_speaker: dict[str, list[tuple[str, dict[str, Any], dict[str, Any], Path]]] = defaultdict(list)
    for bdir in sorted(args.src_dir.iterdir()):
        ann_path = bdir / "annotations.json"
        if not bdir.is_dir() or not ann_path.is_file():
            continue
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        if data.get("stage") != "prepared":
            continue
        items = data.get("items", [])
        if not items:
            continue
        meta = data.get("meta") or {}
        sp_id = str(meta.get("speaker_id") or items[0].get("speaker_id") or "").strip()
        if speaker_filter and sp_id not in speaker_filter:
            continue
        by_speaker[sp_id].append((bdir.name, meta, data, bdir))

    if not by_speaker:
        print(f"[ERROR] No matching prepared batches under {args.src_dir}", file=sys.stderr)
        return 1

    # Pack batches per speaker before touching disk so whisper coverage check is atomic
    emitted_plan: list[tuple[str, str, str, str, list[dict[str, Any]]]] = []
    total_items = 0
    non_empty_whisper = 0

    for sp_id in sorted(by_speaker.keys()):
        src_batches = by_speaker[sp_id]
        if args.max_batches_per_speaker is not None:
            # Sort ascending by item count so fragmented small batches merge first
            src_batches.sort(key=lambda x: (len(x[2].get("items", [])), x[0]))
        else:
            src_batches.sort(key=lambda x: x[0])

        meta0 = src_batches[0][1]
        char_code = str(meta0.get("character_code") or "000")
        sp_ja = str(meta0.get("speaker") or src_batches[0][2]["items"][0].get("speaker") or sp_id)

        flat_entries: list[dict[str, Any]] = []
        for sid, meta, data, bdir in src_batches:
            s_cache = cache_by_story.get(sid, {})
            for it in data.get("items", []):
                iid = str(it["id"])
                wt = s_cache.get(iid)
                if not isinstance(wt, str) or not wt.strip():
                    wt = str(it.get("whisper_text") or "")
                if not wt.strip():
                    wt = str(cache_by_item.get(iid) or "")
                src_wav = bdir / str(it["audio_path"])
                if not src_wav.is_file():
                    print(f"[ERROR] Missing clip wav: {src_wav}", file=sys.stderr)
                    return 1
                flat_entries.append({"src_story": sid, "item": it, "wav": src_wav, "whisper": wt})

        packed = _pack_speaker_batches(flat_entries, args.max_items, args.max_sec, args.max_batches_per_speaker)
        for b_idx, batch in enumerate(packed, start=1):
            batch_id = f"char_{char_code}_{sp_id}_p1_{b_idx:02d}"
            emitted_plan.append((batch_id, sp_id, char_code, sp_ja, batch))
            for e in batch:
                total_items += 1
                if e["whisper"].strip():
                    non_empty_whisper += 1

    ratio = non_empty_whisper / max(1, total_items)
    if ratio < args.require_whisper_ratio:
        print(
            f"[ERROR] Whisper coverage {non_empty_whisper}/{total_items} ({ratio:.1%}) "
            f"< required {args.require_whisper_ratio:.1%}. Aborting without writing.",
            file=sys.stderr,
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    whisper_cache_out: dict[str, dict[str, str]] = {}
    has_fuyuko = False

    for batch_id, sp_id, char_code, sp_ja, batch in emitted_plan:
        if sp_id == "fuyuko":
            has_fuyuko = True
        prof_file = args.profiles_dir / f"{sp_id}.md"
        if not prof_file.is_file():
            print(f"[ERROR] Missing slim profile: {prof_file}", file=sys.stderr)
            return 1
        prof_text = prof_file.read_text(encoding="utf-8").strip()

        bdir = args.out_dir / batch_id
        if bdir.exists():
            if not args.overwrite:
                print(f"[ERROR] Target batch exists (use --overwrite): {bdir}", file=sys.stderr)
                return 1
            shutil.rmtree(bdir)
        clips_dir = bdir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        copied_wavs: list[Path] = []
        new_items: list[dict[str, Any]] = []
        b_cache: dict[str, str] = {}

        for seq, entry in enumerate(batch, start=1):
            orig = entry["item"]
            iid = str(orig["id"])
            clip_name = f"{seq:03d}_{safe_filename(iid)}.wav"
            dst_wav = clips_dir / clip_name
            shutil.copy2(entry["wav"], dst_wav)
            copied_wavs.append(dst_wav)
            b_cache[iid] = entry["whisper"]

            new_items.append({
                "id": iid,
                "seq": seq,
                "line_index": seq - 1,
                "speaker_id": str(orig.get("speaker_id") or sp_id),
                "speaker": str(orig.get("speaker") or sp_ja),
                "text": "",
                "cues": dict(orig.get("cues") or {}),
                "audio_path": f"clips/{clip_name}",
                "source_audio_path": str(orig.get("source_audio_path") or ""),
                "duration": float(orig.get("duration") or 0.0),
                "batch_file": "audio_batch_01.wav",
                "batch_start": 0.0,
                "batch_end": 0.0,
                "pass1": None,
                "pass2": None,
                "human": {"status": "pending", "caption": None, "note": ""},
                "whisper_text": "",
            })

        ts_list = concat_batch_wav(copied_wavs, bdir / "audio_batch_01.wav", args.sample_rate, args.silence_sec)
        for ni, (st, ed) in zip(new_items, ts_list):
            ni["batch_start"] = round(st, 2)
            ni["batch_end"] = round(ed, 2)

        title = f"characters/{char_code}_{sp_id}/repacked_p1"
        ctx_text = (
            f"# Story Context: {batch_id}\n\n"
            f"- **Title**: {title}\n"
            f"- **Target Clips**: {len(new_items)}\n"
            f"- **Script & Audio Manifest**: `script.md`\n"
            f"- **Annotations File**: `annotations.json`\n\n"
            f"## Appearing Character Profiles\n\n"
            f"### {sp_ja} ({sp_id})\n\n"
            f"{prof_text}\n"
        )
        (bdir / "context.md").write_text(ctx_text, encoding="utf-8")

        scr_lines = [
            f"# Story Script & Audio Manifest: {batch_id}",
            "",
            f"- **Title**: {title}",
            f"- **Target Clips**: {len(new_items)}",
            f"- **Concatenated Audio Batches**: `audio_batch_01.wav` (silence gap: `{args.silence_sec}s`)",
            "- **Character Context**: `context.md`",
            "- **Annotations File**: `annotations.json`",
            "",
            "## Script",
            "",
        ]
        for ni in new_items:
            cues = ni["cues"]
            cue_str = ", ".join(f"{k}={v}" for k, v in cues.items() if v not in (None, "", "null"))
            cue_part = f" `[{cue_str}]`" if cue_str else ""
            ts_str = f"{fmt_ts(ni['batch_start'])}–{fmt_ts(ni['batch_end'])}"
            scr_lines.append(
                f"- **[#{ni['seq']:02d} | `{ni['id']}` | `audio_batch_01.wav` {ts_str} ({ni['duration']:.2f}s) | `{ni['audio_path']}`]** "
                f"**{ni['speaker']}**{cue_part}: "
            )
        scr_lines.append("")
        (bdir / "script.md").write_text("\n".join(scr_lines), encoding="utf-8")

        ann_doc = {
            "story_id": batch_id,
            "title": title,
            "meta": {
                "source": "shinycolors_characters",
                "asr_mode": True,
                "repacked": True,
                "character_code": char_code,
                "speaker": sp_ja,
                "speaker_id": sp_id,
                "target_count": len(new_items),
            },
            "stage": "prepared",
            "batches": ["audio_batch_01.wav"],
            "items": new_items,
        }
        (bdir / "annotations.json").write_text(json.dumps(ann_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        whisper_cache_out[batch_id] = b_cache

    cache_out = args.out_dir / ".whisper_cache_p1.json"
    tmp_cache = cache_out.with_suffix(".json.tmp")
    tmp_cache.write_text(json.dumps(whisper_cache_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_cache, cache_out)

    if has_fuyuko:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts/apply_speaker_rules.py"), "--work-dir", str(args.out_dir)],
            check=True,
        )

    print(
        f"[REPACK] emitted_batches={len(emitted_plan)} total_items={total_items} "
        f"whisper_non_empty={non_empty_whisper}/{total_items} ({ratio:.1%}) out_dir={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
