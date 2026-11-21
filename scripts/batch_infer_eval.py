#!/usr/bin/env python3
"""Single-load batch inference helper for monitor_server.py (loads checkpoint once per model)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IRODORI_REPO = Path("/home/darwinwsl/code/irodori/Irodori-TTS")
sys.path.insert(0, str(IRODORI_REPO))

from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--lora-adapter", default=None)
    parser.add_argument("--jobs-json", required=True)
    args = parser.parse_args()

    jobs = json.loads(Path(args.jobs_json).read_text(encoding="utf-8"))
    pending = [j for j in jobs if not Path(j["out_wav"]).exists()]
    if not pending:
        print("All wavs already exist; skipping load.", flush=True)
        return 0

    print(f"[batch_infer] Loading checkpoint once for {len(pending)}/{len(jobs)} wav(s): {args.checkpoint}", flush=True)
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=str(args.checkpoint),
            model_device="cuda",
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision="bf16",
            codec_device="cuda",
            codec_precision="bf16",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=False,
            compile_dynamic=False,
            split_rare_tokens=False,
        )
    )

    for idx, j in enumerate(pending, start=1):
        ref_mode = j["ref_mode"]
        no_ref = ref_mode == "none"
        req = SamplingRequest(
            text=j["text"],
            caption=j.get("caption") or None,
            ref_wav=j.get("ref_wav") if ref_mode == "short" else None,
            ref_wavs=j.get("ref_wavs") if ref_mode == "full" else None,
            no_ref=no_ref,
            seed=42,
            lora_adapter=args.lora_adapter,
        )
        result = runtime.synthesize(req, log_fn=None)
        out_path = save_wav(j["out_wav"], result.audio, result.sample_rate)
        print(f"[{idx}/{len(pending)}] Saved: {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
