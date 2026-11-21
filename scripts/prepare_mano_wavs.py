#!/usr/bin/env python3
import json
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

RAW_DOCS = Path("/home/darwinwsl/code/irodori/datasets/mano/raw_docs.json")
SRC_BASE = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events")
OUT_WAV_DIR = Path("/home/darwinwsl/code/irodori/datasets/mano/wavs")
OUT_JSONL = Path("/home/darwinwsl/code/irodori/datasets/mano/local_dataset.jsonl")

AF_FILTER = (
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:start_silence=0.15,"
    "areverse,"
    "silenceremove=start_periods=1:start_duration=0:start_threshold=-50dB:start_silence=0.15,"
    "areverse"
)


def process_one(item: tuple[int, dict]) -> dict | None:
    idx, doc = item
    src = SRC_BASE / f"{doc['voice']}.m4a"
    dst = OUT_WAV_DIR / f"mano_{idx:05d}.wav"
    if not dst.exists():
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(src),
            "-af", AF_FILTER,
            "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le",
            str(dst),
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0 or not dst.exists():
            return None

    try:
        with wave.open(str(dst), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())
    except Exception:
        return None

    if dur < 2.0 or dur > 15.0:
        dst.unlink(missing_ok=True)
        return None

    return {
        "idx": idx,
        "audio": str(dst),
        "text": doc["text"],
        "speaker_id": "mano",
        "duration": round(dur, 3),
    }


def main():
    OUT_WAV_DIR.mkdir(parents=True, exist_ok=True)
    docs = json.loads(RAW_DOCS.read_text(encoding="utf-8"))
    results = []
    skipped = 0

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(process_one, (i, d)) for i, d in enumerate(docs)]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Converting & trimming"):
            r = fut.result()
            if r is None:
                skipped += 1
            else:
                results.append(r)

    results.sort(key=lambda x: x["idx"])
    total_dur = sum(r["duration"] for r in results)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            row = {
                "audio": r["audio"],
                "text": r["text"],
                "speaker_id": r["speaker_id"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done! Written: {len(results)}, Skipped: {skipped}, Total hours: {total_dur / 3600:.2f}h")


if __name__ == "__main__":
    main()
