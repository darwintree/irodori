#!/usr/bin/env python3
"""Stream raw SC .m4a audio through ffmpeg silence trimming and extract Semantic-DACVAE latents.

Writes latents directly to datasets/shared_latents/<speaker_id>/<sha1(voice_key)[:16]>.pt
and updates latent_path + num_frames in datasets/catalog.sqlite without intermediate WAV dumps.
Must be run with Irodori-TTS/.venv/bin/python (and HF_HUB_OFFLINE=1 if offline).
"""

from __future__ import annotations

import argparse
import hashlib
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sqlite3
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any

import torch

IRODORI_REPO = Path(__file__).resolve().parents[1] / "Irodori-TTS"
if str(IRODORI_REPO) not in sys.path:
    sys.path.insert(0, str(IRODORI_REPO))

from irodori_tts.codec import DACVAECodec

DEFAULT_DB = Path("/home/darwinwsl/code/irodori/datasets/catalog.sqlite")
DEFAULT_OUT_DIR = Path("/home/darwinwsl/code/irodori/datasets/shared_latents")
SAMPLE_RATE = 48000
TRIM_FILTER = (
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:start_silence=0.15,"
    "areverse,"
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:start_silence=0.15,"
    "areverse"
)
_SENTINEL = object()


def latent_target_path(out_dir: Path, speaker_id: str, voice_key: str) -> Path:
    digest = hashlib.sha1(voice_key.encode("utf-8")).hexdigest()[:16]
    return out_dir / speaker_id / f"{digest}.pt"


def decode_trimmed_f32(
    src_path: Path, min_sec: float, max_sec: float
) -> tuple[torch.Tensor | None, str | None]:
    if not src_path.is_file():
        return None, "missing_src"
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(src_path),
        "-af",
        TRIM_FILTER,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        return None, "ffmpeg_err"
    raw = res.stdout
    n_bytes = len(raw) - (len(raw) % 4)
    if n_bytes <= 0:
        return None, "empty_after_trim"
    wav = torch.frombuffer(bytearray(raw[:n_bytes]), dtype=torch.float32)
    dur = wav.numel() / float(SAMPLE_RATE)
    if dur < min_sec:
        return None, "too_short"
    if dur > max_sec:
        return None, "too_long"
    if not bool(torch.isfinite(wav).all()):
        return None, "nonfinite_wav"
    if float(wav.abs().max().item()) < 1e-6:
        return None, "silent"
    return wav, None


def verify_existing_pt(path: Path) -> int | None:
    try:
        t = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(t, torch.Tensor) and t.ndim == 2 and t.shape[1] == 32 and t.shape[0] > 0:
            if bool(torch.isfinite(t).all()):
                return int(t.shape[0])
    except Exception:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract shared DACVAE latents from catalog.sqlite")
    parser.add_argument("--keys", type=Path, required=True, help="selected_voice_keys.txt path")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--codec-repo", default="Aratako/Semantic-DACVAE-Japanese-32dim")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=12, help="Parallel ffmpeg decode workers")
    parser.add_argument("--prefetch", type=int, default=64, help="Bounded decode queue depth")
    parser.add_argument("--commit-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-trimmed-sec", type=float, default=1.0)
    parser.add_argument("--max-trimmed-sec", type=float, default=15.5)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on pending clips to encode")
    args = parser.parse_args()

    keys = [ln.strip() for ln in args.keys.read_text(encoding="utf-8").splitlines() if ln.strip()]
    keys = list(dict.fromkeys(keys))
    if not keys:
        print("[ERROR] No voice keys provided", file=sys.stderr)
        return 1

    conn = sqlite3.connect(str(args.db))
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(clips)").fetchall()}
    if "num_frames" not in existing_cols:
        conn.execute("ALTER TABLE clips ADD COLUMN num_frames INTEGER DEFAULT 0")
        conn.commit()

    # Fetch metadata for all clips once (fast for ~180k rows)
    meta_by_key: dict[str, tuple[str, str, str, int]] = {}
    for vkey, sid, raw_path, lat_path, nframes in conn.execute(
        "SELECT voice_key, speaker_id, raw_m4a_path, COALESCE(latent_path, ''), COALESCE(num_frames, 0) FROM clips"
    ):
        meta_by_key[vkey] = (sid, raw_path, lat_path, int(nframes))

    already_ok = 0
    reused_disk: list[tuple[str, int, str, str]] = []
    pending: list[tuple[str, str, Path, Path]] = []
    missing_db = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    for vkey in keys:
        row = meta_by_key.get(vkey)
        if row is None:
            missing_db += 1
            continue
        sid, raw_path, lat_path, nframes = row
        if lat_path and nframes > 0 and Path(lat_path).is_file():
            already_ok += 1
            continue
        dst = latent_target_path(args.out_dir, sid, vkey)
        if dst.is_file():
            t_frames = verify_existing_pt(dst)
            if t_frames is not None:
                reused_disk.append((str(dst), t_frames, now_iso, vkey))
                continue
        pending.append((vkey, sid, Path(raw_path), dst))

    if reused_disk:
        conn.executemany(
            "UPDATE clips SET latent_path=?, num_frames=?, updated_at=? WHERE voice_key=?",
            reused_disk,
        )
        conn.commit()

    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"[INIT] keys={len(keys)} already_ok={already_ok} reused_disk={len(reused_disk)} "
        f"missing_db={missing_db} to_encode={len(pending)}",
        flush=True,
    )
    if not pending:
        conn.close()
        return 0

    for sid in sorted({sid for _, sid, _, _ in pending}):
        (args.out_dir / sid).mkdir(parents=True, exist_ok=True)

    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=args.device,
        deterministic_encode=True,
        deterministic_decode=True,
        normalize_db=-16.0,
    )
    print(f"[CODEC] Loaded {args.codec_repo} on {args.device} (sr={codec.sample_rate}, dim={codec.latent_dim})", flush=True)

    q: Queue[Any] = Queue(maxsize=max(1, args.prefetch))
    sem = threading.BoundedSemaphore(max(1, args.prefetch))

    def _producer() -> None:
        def _decode_job(item: tuple[str, str, Path, Path]) -> None:
            vkey, sid, src, dst = item
            try:
                wav, err = decode_trimmed_f32(src, args.min_trimmed_sec, args.max_trimmed_sec)
                q.put((vkey, sid, dst, wav, err))
            except Exception:
                q.put((vkey, sid, dst, None, "decode_exc"))

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for item in pending:
                sem.acquire()
                pool.submit(_decode_job, item)
        q.put(_SENTINEL)

    prod_thread = threading.Thread(target=_producer, daemon=True)
    prod_thread.start()

    encoded = 0
    skipped: dict[str, int] = {}
    db_batch: list[tuple[str, int, str, str]] = []
    t0 = time.monotonic()
    seen = 0
    total = len(pending)

    while True:
        pkt = q.get()
        if pkt is _SENTINEL:
            break
        sem.release()
        vkey, sid, dst, wav, err = pkt
        seen += 1

        if err is not None or wav is None:
            key = err or "decode_err"
            skipped[key] = skipped.get(key, 0) + 1
        else:
            try:
                with torch.inference_mode():
                    latent = codec.encode_waveform(wav.unsqueeze(0), sample_rate=SAMPLE_RATE)[0].cpu()
                if latent.ndim != 2 or latent.shape[1] != codec.latent_dim or latent.shape[0] <= 0:
                    raise ValueError(f"bad latent shape {tuple(latent.shape)}")
                if not bool(torch.isfinite(latent).all()):
                    raise ValueError("nonfinite latent")
                tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                torch.save(latent.float().contiguous(), tmp)
                tmp.replace(dst)
                now_str = datetime.now(timezone.utc).isoformat()
                db_batch.append((str(dst), int(latent.shape[0]), now_str, vkey))
                encoded += 1
            except Exception:
                skipped["encode_err"] = skipped.get("encode_err", 0) + 1

        if len(db_batch) >= args.commit_every:
            conn.executemany(
                "UPDATE clips SET latent_path=?, num_frames=?, updated_at=? WHERE voice_key=?",
                db_batch,
            )
            conn.commit()
            db_batch.clear()

        if seen % args.log_every == 0 or seen == total:
            elapsed = max(time.monotonic() - t0, 1e-6)
            rate = seen / elapsed
            eta_min = (total - seen) / rate / 60.0 if rate > 0 else 0.0
            print(
                f"[PROGRESS] {seen}/{total} ({seen/total*100:.1f}%) | encoded={encoded} | "
                f"skipped={sum(skipped.values())} {skipped} | {rate:.1f} utt/s | ETA={eta_min:.1f}m",
                flush=True,
            )

    if db_batch:
        conn.executemany(
            "UPDATE clips SET latent_path=?, num_frames=?, updated_at=? WHERE voice_key=?",
            db_batch,
        )
        conn.commit()

    prod_thread.join()
    conn.close()
    elapsed = time.monotonic() - t0
    print(f"[DONE] encoded={encoded}/{total} in {elapsed/60:.1f}m | skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
