#!/usr/bin/env python3
"""Fixed-subcommand runner for SC29 pipeline steps that require host mounts, conda envs, or HF cache.

Only permits hardcoded, audited subcommands via subprocess(..., shell=False) or direct module calls.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/home/darwinwsl/code/irodori")
VENV_PY = ROOT / "Irodori-TTS/.venv/bin/python"
WHISPER_PY = Path("/home/darwinwsl/miniconda3/envs/whisperx/bin/python")
SC_CHARS_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/characters")
SC_EVENTS_ROOT = Path("/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events")
HF_CACHE = Path.home() / ".cache/huggingface/hub"

CHAR_SOURCES_DIR = ROOT / "datasets/annotator_sources/characters"
CHAR_WORK_DIR = ROOT / "datasets/annotator_work/characters"
CHARS_PROFILE_DIR = ROOT / "references/eidolon-sc/characters/facts"

TRAIN_OUTPUT_DIR = ROOT / "outputs/sc29_v1_full"
TRAIN_CONFIG = ROOT / "configs/train_sc29_v1_full.yaml"
TRAIN_MANIFEST = ROOT / "datasets/sc29_v1/train_manifest.jsonl"
INIT_CKPT = ROOT / "models/Irodori-TTS-v4.1-Small/model.safetensors"
TOKENIZER_SRC = ROOT / "models/Irodori-TTS-v4.1-Small/tokenizer"


def cmd_check(_: argparse.Namespace) -> int:
    ok = True
    checks = [
        ("SC char voices mount", SC_CHARS_ROOT.is_dir()),
        ("SC event voices mount", SC_EVENTS_ROOT.is_dir()),
        ("whisperx python", WHISPER_PY.is_file()),
        ("Irodori-TTS venv python", VENV_PY.is_file()),
        ("Base model safetensors", INIT_CKPT.is_file()),
        ("DACVAE HF cache", (HF_CACHE / "models--Aratako--Semantic-DACVAE-Japanese-32dim").is_dir()),
        ("ModernBERT HF cache", any(HF_CACHE.glob("*modernbert*")) or (TOKENIZER_SRC).is_dir()),
    ]
    for label, passed in checks:
        print(f"[CHECK] {label}: {'OK' if passed else 'MISSING'}")
        if not passed:
            ok = False

    if VENV_PY.is_file():
        probe = subprocess.run(
            [
                str(VENV_PY),
                "-c",
                "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
            ],
            capture_output=True,
            text=True,
        )
        print(f"[CHECK] CUDA probe (rc={probe.returncode}): {probe.stdout.strip() or probe.stderr.strip()}")
        if probe.returncode != 0 or "True" not in probe.stdout:
            ok = False

    usage = shutil.disk_usage(ROOT)
    print(f"[CHECK] Disk free: {usage.free / (1024**3):.1f} GiB / {usage.total / (1024**3):.1f} GiB")
    return 0 if ok else 1


def cmd_prepare_p1(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(ROOT / "voice-caption-annotator"))
    from prepare import prepare_story

    sources = sorted(CHAR_SOURCES_DIR.glob("*.json"))
    pending = [s for s in sources if not (CHAR_WORK_DIR / s.stem / "annotations.json").is_file()]
    print(f"[PREPARE-P1] total_sources={len(sources)} pending={len(pending)} workers={args.workers}", flush=True)
    if not pending:
        return 0

    CHAR_WORK_DIR.mkdir(parents=True, exist_ok=True)
    done = 0
    failed = []

    def _run_one(src: Path) -> tuple[str, bool, str]:
        try:
            prepare_story(
                source_path=src,
                work_root=CHAR_WORK_DIR,
                characters_dir=CHARS_PROFILE_DIR,
                batch_size=20,
                max_batch_sec=180.0,
                silence_sec=0.8,
                sample_rate=48000,
                trim_silence=True,
                overwrite=False,
            )
            return src.stem, True, "ok"
        except Exception as exc:
            return src.stem, False, str(exc)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = [pool.submit(_run_one, src) for src in pending]
        for fut in as_completed(futs):
            name, ok, msg = fut.result()
            if ok:
                done += 1
                if done % 50 == 0 or done == len(pending):
                    print(f"  prepared {done}/{len(pending)}", flush=True)
            else:
                failed.append(f"{name} ({msg})")

    print(f"[PREPARE-P1] ok={done}/{len(pending)} failed={len(failed)}", flush=True)
    if failed:
        print(f"[PREPARE-P1 FAILURES] {failed}", file=sys.stderr)
        return 1
    return 0


def cmd_extract_latents(args: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "scripts/extract_shared_latents.py"),
        "--keys",
        str(ROOT / "datasets/sc29_v1/selected_voice_keys.txt"),
        "--db",
        str(ROOT / "datasets/catalog.sqlite"),
        "--out-dir",
        str(ROOT / "datasets/shared_latents"),
        "--workers",
        str(args.workers),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    return subprocess.run(cmd, env=env).returncode


def cmd_verify_manifest(_: argparse.Namespace) -> int:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    code = (
        "import sys; sys.path.insert(0, '/home/darwinwsl/code/irodori/Irodori-TTS'); "
        "from irodori_tts.dataset import LatentTextDataset; "
        "ds = LatentTextDataset('/home/darwinwsl/code/irodori/datasets/sc29_v1/train_manifest.jsonl', latent_dim=32, max_latent_steps=500, enable_caption_condition=True, enable_speaker_condition=True, ref_min_frames=25, ref_max_frames=750); "
        "vds = LatentTextDataset('/home/darwinwsl/code/irodori/datasets/sc29_v1/valid_manifest.jsonl', latent_dim=32, max_latent_steps=500, enable_caption_condition=True, enable_speaker_condition=True, ref_min_frames=25, ref_max_frames=750); "
        "print('train_len=', len(ds), 'valid_len=', len(vds), 'speakers=', len(ds.manifest_index.speakers)); "
        "for i in range(min(5, len(ds))): s=ds[i]; assert s['latent'].ndim==2 and s['latent'].shape[1]==32 and s['num_frames']>0 and s['text']; "
        "print('sample[0]=', ds[0]['latent'].shape, ds[0]['ref_latent'].shape, repr(ds[0]['caption'][:40]))"
    )
    return subprocess.run([str(VENV_PY), "-c", code], env=env).returncode


def cmd_train(args: argparse.Namespace) -> int:
    TRAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tok_link = TRAIN_OUTPUT_DIR / "tokenizer"
    if not tok_link.exists() and not tok_link.is_symlink():
        tok_link.symlink_to(TOKENIZER_SRC)

    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "Irodori-TTS/train.py"),
        "--config",
        str(TRAIN_CONFIG),
        "--manifest",
        str(TRAIN_MANIFEST),
        "--output-dir",
        str(TRAIN_OUTPUT_DIR),
    ]
    if args.resume:
        cmd += ["--resume", str(args.resume)]
    else:
        cmd += ["--init-checkpoint", str(INIT_CKPT)]
    return subprocess.run(cmd, env=env).returncode


def cmd_convert_best(args: argparse.Namespace) -> int:
    in_path = Path(args.input).resolve()
    out_dir_resolved = TRAIN_OUTPUT_DIR.resolve()
    if os.path.commonpath([str(in_path), str(out_dir_resolved)]) != str(out_dir_resolved):
        raise ValueError(f"--input must be inside {TRAIN_OUTPUT_DIR}: got {in_path}")
    if not in_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")

    out_path = Path(args.output).resolve()
    if os.path.commonpath([str(out_path), str(out_dir_resolved)]) != str(out_dir_resolved):
        raise ValueError(f"--output must be inside {TRAIN_OUTPUT_DIR}: got {out_path}")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "Irodori-TTS/convert_checkpoint_to_safetensors.py"),
        str(in_path),
        "--output",
        str(out_path),
        "--force",
    ]
    return subprocess.run(cmd, env=env).returncode


def cmd_eval(args: argparse.Namespace) -> int:
    eval_py = ROOT / "scripts/eval_sc29.py"
    if not eval_py.is_file():
        raise FileNotFoundError(f"Missing evaluation script: {eval_py}")
    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [str(VENV_PY), str(eval_py)]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.port:
        cmd += ["--port", str(args.port)]
    return subprocess.run(cmd, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Audited single-entrypoint runner for SC29 pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("check", help="Run read-only pre-flight checks")

    p_p1 = sub.add_parser("prepare-p1", help="Prepare pending Wave A-P1+P2 character batches")
    p_p1.add_argument("--workers", type=int, default=12)

    p_lat = sub.add_parser("extract-latents", help="Extract shared DACVAE latents into shared_latents/")
    p_lat.add_argument("--workers", type=int, default=12)
    p_lat.add_argument("--limit", type=int, default=None)

    sub.add_parser("verify-manifest", help="Verify LatentTextDataset loads sc29_v1 manifests")

    p_trn = sub.add_parser("train", help="Run 29-character Full FT training")
    p_trn.add_argument("--resume", default=None)

    p_cvt = sub.add_parser("convert-best", help="Convert best .pt checkpoint inside outputs/sc29_v1_full/ to safetensors")
    p_cvt.add_argument("--input", required=True)
    p_cvt.add_argument("--output", default=str(TRAIN_OUTPUT_DIR / "model_best.safetensors"))

    p_ev = sub.add_parser("eval", help="Run multi-character and Fuyuko dual-persona evaluation")
    p_ev.add_argument("--checkpoint", default=None)
    p_ev.add_argument("--out-dir", default=None)
    p_ev.add_argument("--port", type=int, default=None)

    args = parser.parse_args()
    dispatch = {
        "check": cmd_check,
        "prepare-p1": cmd_prepare_p1,
        "extract-latents": cmd_extract_latents,
        "verify-manifest": cmd_verify_manifest,
        "train": cmd_train,
        "convert-best": cmd_convert_best,
        "eval": cmd_eval,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    sys.exit(main())
