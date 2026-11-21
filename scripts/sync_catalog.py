#!/usr/bin/env python3
"""Synchronize Shiny Colors voice assets and voice-caption-annotator outputs into datasets/catalog.sqlite.

Single source of truth for:
1. Event voices (2-15s) from MongoDB shinycolors.voices
2. Character voices (2-15s, P0+P1 for all 28 idols + P2 for low-count idols + Hazuki 901)
3. Completed annotations (Pass 1 / Whisper / Pass 2 / Human) from annotator work directories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DB = Path("/home/darwinwsl/code/irodori/datasets/catalog.sqlite")
DEFAULT_EVENT_VOICE_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events")
DEFAULT_CHAR_VOICE_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/characters")
DEFAULT_SC_NODE_MODULES = Path("/mnt/p/sc/sc-asset-map-manager/node_modules")

SPEAKERS: list[tuple[str, str, str]] = [
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
    ("901", "はづき", "hazuki"),
]
JA_TO_SID = {ja: sid for _, ja, sid in SPEAKERS}
LOW_COUNT_SIDS = {"hana", "luca", "madoka", "haruki", "mikoto", "toru", "hazuki"}

EMO_WEIGHTS = {
    "cry": 5, "anger": 4, "sad": 3, "surp": 3, "shy": 2,
    "serious": 2, "trouble": 2, "down": 3, "sigh": 2,
    "doya": 2, "jito": 2, "excite": 2, "joy": 1,
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS clips (
    voice_key TEXT PRIMARY KEY,
    speaker_id TEXT NOT NULL,
    speaker_ja TEXT NOT NULL,
    source_type TEXT NOT NULL,       -- 'event' | 'character'
    category TEXT NOT NULL,          -- e.g. produce_events, casting, talking_multi
    story_or_batch_id TEXT DEFAULT '',
    priority TEXT DEFAULT '',
    raw_m4a_path TEXT NOT NULL,
    wav_path TEXT DEFAULT '',
    latent_path TEXT DEFAULT '',
    num_frames INTEGER DEFAULT 0,
    duration REAL NOT NULL,
    emo_score INTEGER DEFAULT 0,
    char_anim1 TEXT DEFAULT '',
    char_anim2 TEXT DEFAULT '',
    text_raw TEXT DEFAULT '',
    whisper_text TEXT DEFAULT '',
    text_final TEXT DEFAULT '',
    text_source TEXT DEFAULT '',
    text_hash TEXT DEFAULT '',
    caption_ja TEXT DEFAULT '',
    caption_zh TEXT DEFAULT '',
    caption_stage TEXT DEFAULT 'none', -- none | pass1_done | reviewed | human_approved | rejected
    flags TEXT DEFAULT '[]',
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_speaker ON clips(speaker_id);
CREATE INDEX IF NOT EXISTS idx_clips_stage ON clips(caption_stage);
CREATE INDEX IF NOT EXISTS idx_clips_source ON clips(source_type, category);
CREATE INDEX IF NOT EXISTS idx_clips_story ON clips(story_or_batch_id);
"""


def norm_text_hash(speaker_id: str, text: str) -> str:
    norm = re.sub(r"\s+", "", (text or "").replace("\r\n", "\n")).strip()
    if not norm:
        return ""
    return hashlib.sha1(f"{speaker_id}::{norm}".encode("utf-8")).hexdigest()[:16]


def compute_emo_score(anim1: str, anim2: str) -> int:
    s = f"{anim1 or ''} {anim2 or ''}".lower()
    score = 0
    for kw, w in EMO_WEIGHTS.items():
        if kw in s:
            score += w
    return score


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


def classify_char_voice(filename: str) -> tuple[str, str] | None:
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
    if re.search(r"(official_sample_voice|^mypage_|excellent|good|normal|bad|failure|success|win|lose|clear|retry|cutin)", filename, re.I):
        return ("P2", "mypage_or_result")
    return ("P3", "system_misc")


def sync_mongo_events(conn: sqlite3.Connection, min_dur: float, max_dur: float) -> int:
    node_code = f"""
const {{ MongoClient }} = require({json.dumps(str(DEFAULT_SC_NODE_MODULES / 'mongodb'))});
const speakers = {json.dumps(list(JA_TO_SID.keys()))};
(async () => {{
  const c = new MongoClient('mongodb://127.0.0.1:27017');
  await c.connect();
  const col = c.db('shinycolors').collection('voices');
  const cur = col.find(
    {{ speaker: {{ $in: speakers }}, duration: {{ $gte: {min_dur}, $lte: {max_dur} }}, voice: {{ $exists: true, $ne: '' }}, text: {{ $exists: true, $ne: '' }} }},
    {{ projection: {{ speaker: 1, voice: 1, text: 1, duration: 1, charAnim1: 1, charAnim2: 1 }} }}
  );
  for await (const d of cur) {{
    process.stdout.write(JSON.stringify(d) + '\\n');
  }}
  await c.close();
}})().catch(e => {{ console.error(e); process.exit(1); }});
"""
    proc = subprocess.Popen(["node", "-e", node_code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    now = datetime.now(timezone.utc).isoformat()
    batch: list[tuple[Any, ...]] = []
    count = 0

    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        doc = json.loads(line)
        ja = doc["speaker"]
        sid = JA_TO_SID.get(ja)
        if not sid:
            continue
        voice = str(doc["voice"]).strip()
        parts = voice.split("/")
        category = parts[0] if parts else "event"
        story_id = parts[1] if len(parts) > 2 else ""
        text = str(doc["text"]).replace("\r\n", "\n").strip()
        a1 = str(doc.get("charAnim1") or "")
        a2 = str(doc.get("charAnim2") or "")
        emo = compute_emo_score(a1, a2)
        vkey = f"events/{voice}"
        raw_path = str(DEFAULT_EVENT_VOICE_ROOT / f"{voice}.m4a")

        batch.append((
            vkey, sid, ja, "event", category, story_id, "",
            raw_path, float(doc["duration"]), emo, a1, a2,
            text, text, "sc_json", norm_text_hash(sid, text), now,
        ))
        count += 1
        if len(batch) >= 2000:
            _upsert_event_batch(conn, batch)
            batch.clear()

    if batch:
        _upsert_event_batch(conn, batch)
    rc = proc.wait()
    if rc != 0:
        err = proc.stderr.read() if proc.stderr else ""
        raise RuntimeError(f"MongoDB export failed (rc={rc}): {err}")
    conn.commit()
    return count


def _upsert_event_batch(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO clips (
            voice_key, speaker_id, speaker_ja, source_type, category, story_or_batch_id, priority,
            raw_m4a_path, duration, emo_score, char_anim1, char_anim2,
            text_raw, text_final, text_source, text_hash, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(voice_key) DO UPDATE SET
            duration = excluded.duration,
            emo_score = excluded.emo_score,
            char_anim1 = excluded.char_anim1,
            char_anim2 = excluded.char_anim2,
            text_raw = excluded.text_raw,
            text_final = CASE WHEN clips.text_final = '' THEN excluded.text_final ELSE clips.text_final END,
            text_source = CASE WHEN clips.text_source = '' THEN excluded.text_source ELSE clips.text_source END,
            text_hash = CASE WHEN clips.text_hash = '' THEN excluded.text_hash ELSE clips.text_hash END,
            updated_at = excluded.updated_at
        """,
        batch,
    )


def sync_char_voices(conn: sqlite3.Connection, min_dur: float, max_dur: float) -> int:
    now = datetime.now(timezone.utc).isoformat()
    batch: list[tuple[Any, ...]] = []
    count = 0
    for code, ja, sid in SPEAKERS:
        cdir = DEFAULT_CHAR_VOICE_ROOT / code
        if not cdir.is_dir():
            continue
        for fp in sorted(cdir.glob("*.m4a")):
            cls = classify_char_voice(fp.name)
            if cls is None:
                continue
            prio, bucket = cls
            # Include P0+P1 for all idols; also include P2 for low-count speakers
            if prio == "P3":
                continue
            if prio == "P2" and sid not in LOW_COUNT_SIDS:
                continue
            dur = parse_mp4_duration(fp)
            if dur is None or dur < min_dur or dur > max_dur:
                continue
            vkey = f"characters/{code}/{fp.stem}"
            emo = 5 if bucket in {"casting", "emotion_tag"} else (3 if prio == "P1" else 1)
            batch.append((
                vkey, sid, ja, "character", bucket, "", prio,
                str(fp), round(dur, 3), emo, now,
            ))
            count += 1
            if len(batch) >= 1000:
                _upsert_char_batch(conn, batch)
                batch.clear()
    if batch:
        _upsert_char_batch(conn, batch)
    conn.commit()
    return count


def _upsert_char_batch(conn: sqlite3.Connection, batch: list[tuple[Any, ...]]) -> None:
    conn.executemany(
        """
        INSERT INTO clips (
            voice_key, speaker_id, speaker_ja, source_type, category, story_or_batch_id, priority,
            raw_m4a_path, duration, emo_score, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(voice_key) DO UPDATE SET
            priority = excluded.priority,
            category = excluded.category,
            duration = excluded.duration,
            updated_at = excluded.updated_at
        """,
        batch,
    )


def derive_voice_key(source_audio_path: str) -> str | None:
    p = source_audio_path.replace("\\", "/")
    if "/sounds/voice/events/" in p:
        rel = p.split("/sounds/voice/events/", 1)[1]
        if rel.endswith(".m4a"):
            rel = rel[:-4]
        return f"events/{rel}"
    if "/sounds/voice/characters/" in p:
        rel = p.split("/sounds/voice/characters/", 1)[1]
        if rel.endswith(".m4a"):
            rel = rel[:-4]
        return f"characters/{rel}"
    return None


def sync_annotations(conn: sqlite3.Connection, work_dirs: list[Path]) -> tuple[int, int]:
    now = datetime.now(timezone.utc).isoformat()
    updated = 0
    skipped = 0

    for wdir in work_dirs:
        if not wdir.is_dir():
            continue
        for ann_path in sorted(wdir.glob("*/annotations.json")):
            try:
                data = json.loads(ann_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            story_id = data.get("story_id", ann_path.parent.name)
            top_stage = data.get("stage", "prepared")
            for it in data.get("items", []):
                vkey = derive_voice_key(str(it.get("source_audio_path") or ""))
                if not vkey:
                    skipped += 1
                    continue
                p1 = it.get("pass1") or {}
                p2 = it.get("pass2") or {}
                hum = it.get("human") or {}

                # Determine effective stage & caption
                h_st = hum.get("status", "pending")
                p2_st = p2.get("status", "")
                if h_st == "rejected" or p2_st == "rejected":
                    stage = "rejected"
                elif h_st == "approved":
                    stage = "human_approved"
                elif p2_st in {"approved", "revised", "needs_human"}:
                    stage = "reviewed"
                elif p1.get("caption"):
                    stage = "pass1_done"
                else:
                    stage = "none"

                cap_ja = (hum.get("caption") or p2.get("caption") or p1.get("caption") or "").strip()
                cap_zh = (p2.get("caption_zh") or p1.get("caption_zh") or "").strip()
                whisper_txt = (it.get("whisper_text") or "").strip()
                final_txt = (p2.get("text") or it.get("text") or p1.get("transcript") or "").strip()
                txt_src = p2.get("text_source") or ("sc_json" if vkey.startswith("events/") else "")
                flags = json.dumps(p1.get("flags") or [], ensure_ascii=False)
                sid = str(it.get("speaker_id") or "")
                thash = norm_text_hash(sid, final_txt) if (sid and final_txt) else ""

                # Protect existing human_approved from being downgraded by automated runs
                cur = conn.execute(
                    "SELECT caption_stage, speaker_id FROM clips WHERE voice_key = ?",
                    (vkey,),
                ).fetchone()
                if cur is None:
                    skipped += 1
                    continue
                if cur[0] == "human_approved" and stage != "human_approved":
                    continue
                if not thash and final_txt:
                    thash = norm_text_hash(cur[1], final_txt)

                conn.execute(
                    """
                    UPDATE clips SET
                        story_or_batch_id = CASE WHEN ? != '' THEN ? ELSE story_or_batch_id END,
                        whisper_text = CASE WHEN ? != '' THEN ? ELSE whisper_text END,
                        text_final = CASE WHEN ? != '' THEN ? ELSE text_final END,
                        text_source = CASE WHEN ? != '' THEN ? ELSE text_source END,
                        text_hash = CASE WHEN ? != '' THEN ? ELSE text_hash END,
                        caption_ja = CASE WHEN ? != '' THEN ? ELSE caption_ja END,
                        caption_zh = CASE WHEN ? != '' THEN ? ELSE caption_zh END,
                        caption_stage = ?,
                        flags = ?,
                        updated_at = ?
                    WHERE voice_key = ?
                    """,
                    (
                        story_id, story_id,
                        whisper_txt, whisper_txt,
                        final_txt, final_txt,
                        txt_src, txt_src,
                        thash, thash,
                        cap_ja, cap_ja,
                        cap_zh, cap_zh,
                        stage, flags, now, vkey,
                    ),
                )
                updated += 1
    conn.commit()
    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync SC voices & annotations into catalog.sqlite")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--skip-mongo", action="store_true", help="Skip syncing events from MongoDB")
    parser.add_argument("--skip-chars", action="store_true", help="Skip scanning characters/ m4a files")
    parser.add_argument("--work-dir", action="append", type=Path, default=[], help="Annotator work dir(s) to ingest")
    parser.add_argument("--min-dur", type=float, default=2.0)
    parser.add_argument("--max-dur", type=float, default=15.0)
    args = parser.parse_args()

    args.db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(args.db))
    conn.executescript(SCHEMA_SQL)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)").fetchall()}
    if "num_frames" not in existing_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN num_frames INTEGER DEFAULT 0")
        conn.commit()

    if not args.skip_mongo:
        n_ev = sync_mongo_events(conn, args.min_dur, args.max_dur)
        print(f"[SYNC] MongoDB events (2-15s): {n_ev} rows upserted", flush=True)

    if not args.skip_chars:
        n_ch = sync_char_voices(conn, args.min_dur, args.max_dur)
        print(f"[SYNC] Character voices (P0+P1 + low-count P2): {n_ch} rows upserted", flush=True)

    work_dirs = args.work_dir or [
        Path("/home/darwinwsl/code/irodori/datasets/annotator_work/events"),
        Path("/home/darwinwsl/code/irodori/datasets/annotator_work/characters"),
    ]
    n_upd, n_skip = sync_annotations(conn, work_dirs)
    print(f"[SYNC] Annotations ingested: updated={n_upd}, skipped={n_skip}", flush=True)

    # Print quick catalog summary
    stats = conn.execute(
        """
        SELECT source_type, caption_stage, COUNT(*)
        FROM clips
        GROUP BY source_type, caption_stage
        ORDER BY source_type, caption_stage
        """
    ).fetchall()
    print("[CATALOG SUMMARY]")
    for stype, stage, cnt in stats:
        print(f"  {stype:10s} | {stage:15s} | {cnt}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
