#!/usr/bin/env python3
"""Fixed-subcommand runner for SC29 pipeline steps that require host mounts, conda envs, or HF cache.

Only permits hardcoded, audited subcommands via subprocess(..., shell=False) or direct module calls.
"""

from __future__ import annotations

import argparse
import json
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
TRAIN_CONFIG_LITE = ROOT / "configs/train_sc29_v1_lite.yaml"
TRAIN_MANIFEST = ROOT / "datasets/sc29_v1/train_manifest.jsonl"
INIT_CKPT = ROOT / "models/Irodori-TTS-v4.1-Small/model.safetensors"
TOKENIZER_SRC = ROOT / "models/Irodori-TTS-v4.1-Small/tokenizer"
SPEAKER_INV_CONFIG = ROOT / "configs/train_sc29_speaker_inversion.yaml"
MEANFLOW_4090_CONFIG = ROOT / "configs/train_sc29_meanflow_4090.yaml"


def _variant_paths(variant: str) -> tuple[Path, Path]:
    if variant == "lite":
        return ROOT / "datasets/sc29_v1_lite", ROOT / "outputs/sc29_v1_lite"
    return ROOT / "datasets/sc29_v1", ROOT / "outputs/sc29_v1_full"


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


def cmd_whisper_p1(args: argparse.Namespace) -> int:
    if not CHAR_WORK_DIR.is_dir():
        print(f"[WHISPER-P1] Work dir not found: {CHAR_WORK_DIR}", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"

    if args.benchmark:
        cmd = [
            sys.executable,
            str(ROOT / "voice-caption-annotator/whisper_transcribe.py"),
            "--work-dir",
            str(CHAR_WORK_DIR),
            "--whisper-python",
            str(WHISPER_PY),
            "--local-files-only",
            "--batch-size",
            str(args.batch_size),
            "--benchmark-compare",
            "--story",
            "char_004_kogane_live_skill_01",
            "--story",
            "char_018_fuyuko_live_skill_01",
        ]
        return subprocess.run(cmd, env=env).returncode

    pending: list[Path] = []
    for d in sorted(CHAR_WORK_DIR.iterdir()):
        ann = d / "annotations.json"
        if not d.is_dir() or not ann.is_file():
            continue
        data = json.loads(ann.read_text(encoding="utf-8"))
        if data.get("stage") == "prepared":
            pending.append(d)

    print(f"[WHISPER-P1] pending prepared batches={len(pending)}", flush=True)
    if not pending:
        return 0

    cmd = [
        sys.executable,
        str(ROOT / "voice-caption-annotator/whisper_transcribe.py"),
        "--work-dir",
        str(CHAR_WORK_DIR),
        "--whisper-python",
        str(WHISPER_PY),
        "--local-files-only",
        "--batch-size",
        str(args.batch_size),
    ]
    if args.force:
        cmd.append("--force")
    for d in pending:
        cmd += ["--story", d.name]

    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        return rc

    if not args.no_stash:
        cache_file = CHAR_WORK_DIR / ".whisper_cache_p1.json"
        if args.force and cache_file.is_file():
            cache_file.unlink()
        stash_cmd = [
            sys.executable,
            str(ROOT / "scripts/whisper_cache_ctl.py"),
            "stash",
            "--work-dir",
            str(CHAR_WORK_DIR),
        ]
        return subprocess.run(stash_cmd).returncode
    return 0


def cmd_extract_latents(args: argparse.Namespace) -> int:
    dataset_dir, _ = _variant_paths(args.variant)
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "scripts/extract_shared_latents.py"),
        "--keys",
        str(dataset_dir / "selected_voice_keys.txt"),
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


def cmd_verify_manifest(args: argparse.Namespace) -> int:
    dataset_dir, _ = _variant_paths(args.variant)
    train_manifest = str(dataset_dir / "train_manifest.jsonl")
    valid_manifest = str(dataset_dir / "valid_manifest.jsonl")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    code = "\n".join([
        "import sys",
        "sys.path.insert(0, '/home/darwinwsl/code/irodori/Irodori-TTS')",
        "from irodori_tts.dataset import LatentTextDataset",
        "ds = LatentTextDataset(sys.argv[1], latent_dim=32, max_latent_steps=500, enable_caption_condition=True, enable_speaker_condition=True, ref_min_frames=25, ref_max_frames=750)",
        "vds = LatentTextDataset(sys.argv[2], latent_dim=32, max_latent_steps=500, enable_caption_condition=True, enable_speaker_condition=True, ref_min_frames=25, ref_max_frames=750)",
        "print('train_len=', len(ds), 'valid_len=', len(vds), 'speakers=', len(ds.manifest_index.speakers))",
        "for i in range(min(5, len(ds))):",
        "    s = ds[i]",
        "    assert s['latent'].ndim == 2 and s['latent'].shape[1] == 32 and s['num_frames'] > 0 and s['text']",
        "print('sample[0]=', ds[0]['latent'].shape, ds[0]['ref_latent'].shape, repr(ds[0]['caption'][:40]))",
    ])
    return subprocess.run([str(VENV_PY), "-c", code, train_manifest, valid_manifest], env=env).returncode


def cmd_train(args: argparse.Namespace) -> int:
    dataset_dir, output_dir = _variant_paths(args.variant)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_link = output_dir / "tokenizer"
    if not tok_link.exists() and not tok_link.is_symlink():
        tok_link.symlink_to(TOKENIZER_SRC)

    train_cfg = TRAIN_CONFIG_LITE if (args.variant == "lite" and TRAIN_CONFIG_LITE.is_file()) else TRAIN_CONFIG
    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "Irodori-TTS/train.py"),
        "--config",
        str(train_cfg),
        "--manifest",
        str(dataset_dir / "train_manifest.jsonl"),
        "--output-dir",
        str(output_dir),
    ]
    if args.resume:
        cmd += ["--resume", str(args.resume)]
    else:
        cmd += ["--init-checkpoint", str(INIT_CKPT)]
    return subprocess.run(cmd, env=env).returncode


def cmd_convert_best(args: argparse.Namespace) -> int:
    _, output_dir = _variant_paths(args.variant)
    in_path = Path(args.input).resolve()
    out_dir_resolved = output_dir.resolve()
    if os.path.commonpath([str(in_path), str(out_dir_resolved)]) != str(out_dir_resolved):
        raise ValueError(f"--input must be inside {output_dir}: got {in_path}")
    if not in_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {in_path}")

    out_target = Path(args.output) if args.output else (output_dir / "model_best.safetensors")
    out_path = out_target.resolve()
    if os.path.commonpath([str(out_path), str(out_dir_resolved)]) != str(out_dir_resolved):
        raise ValueError(f"--output must be inside {output_dir}: got {out_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok_link = out_path.parent / "tokenizer"
    if not tok_link.exists() and not tok_link.is_symlink():
        tok_link.symlink_to(TOKENIZER_SRC)

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(ROOT / 'Irodori-TTS')!r})",
        "import convert_checkpoint_to_safetensors as c",
        "c._export_tokenizer = lambda flat_config, output_path, *, source_checkpoint=None: (output_path.parent / 'tokenizer')",
        f"sys.argv = ['convert_checkpoint_to_safetensors.py', {str(in_path)!r}, '--output', {str(out_path)!r}, '--force']",
        "c.main()",
    ])
    cmd = [str(VENV_PY), "-c", code]
    return subprocess.run(cmd, env=env).returncode


def cmd_eval(args: argparse.Namespace) -> int:
    eval_py = ROOT / "scripts/eval_sc29.py"
    if not eval_py.is_file():
        raise FileNotFoundError(f"Missing evaluation script: {eval_py}")
    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [str(VENV_PY), str(eval_py)]
    if args.base_checkpoint:
        cmd += ["--base-checkpoint", str(args.base_checkpoint)]
    if args.checkpoint:
        cmd += ["--checkpoint", str(args.checkpoint)]
    for cp in args.extra_checkpoint:
        cmd += ["--extra-checkpoint", str(cp)]
    if args.speakers:
        cmd += ["--speakers", str(args.speakers)]
    if args.out_dir:
        cmd += ["--out-dir", str(args.out_dir)]
    if args.port:
        cmd += ["--port", str(args.port)]
    if args.serve_only:
        cmd.append("--serve-only")
    return subprocess.run(cmd, env=env).returncode


def cmd_invert_speaker(args: argparse.Namespace) -> int:
    import re

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(f"Full train manifest not found: {TRAIN_MANIFEST}")
    if not SPEAKER_INV_CONFIG.is_file():
        raise FileNotFoundError(f"Speaker inversion config not found: {SPEAKER_INV_CONFIG}")

    speakers = [str(s).strip() for s in args.speakers if str(s).strip()]
    if not speakers:
        raise ValueError("--speakers must specify at least one speaker ID")

    out_root = Path(args.out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifests_root = ROOT / "datasets/sc29_v1_speakers"
    manifests_root.mkdir(parents=True, exist_ok=True)

    rows_by_sp = {sp: [] for sp in speakers}
    seen_keys = {sp: set() for sp in speakers}
    target_set = set(speakers)
    with TRAIN_MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sp = str(row.get("speaker_id") or "")
            if sp not in target_set:
                continue
            dedup_key = str(row.get("voice_key") or row.get("latent_path") or "")
            if dedup_key and dedup_key in seen_keys[sp]:
                continue
            if dedup_key:
                seen_keys[sp].add(dedup_key)
            rows_by_sp[sp].append(line if line.endswith("\n") else line + "\n")

    for sp in speakers:
        if not rows_by_sp[sp]:
            raise ValueError(f"Speaker {sp!r} has 0 rows in {TRAIN_MANIFEST}")

    best_re = re.compile(
        r"^checkpoint_best_val_loss_(\d+)_([0-9]+(?:\.[0-9]+)?)\.speaker\.safetensors$"
    )
    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"

    for idx, sp in enumerate(speakers, start=1):
        sp_manifest = manifests_root / sp / "train_manifest.jsonl"
        sp_manifest.parent.mkdir(parents=True, exist_ok=True)
        sp_manifest.write_text("".join(rows_by_sp[sp]), encoding="utf-8")

        sp_out_dir = out_root / sp
        sp_out_dir.mkdir(parents=True, exist_ok=True)
        tok_link = sp_out_dir / "tokenizer"
        if not tok_link.exists() and not tok_link.is_symlink():
            tok_link.symlink_to(TOKENIZER_SRC)

        print(
            f"\n[INVERT {idx}/{len(speakers)}] speaker={sp} rows={len(rows_by_sp[sp])} max_steps={args.max_steps}",
            flush=True,
        )
        cmd = [
            str(VENV_PY),
            str(ROOT / "Irodori-TTS/train.py"),
            "--config",
            str(SPEAKER_INV_CONFIG),
            "--manifest",
            str(sp_manifest),
            "--init-checkpoint",
            str(ckpt_path),
            "--output-dir",
            str(sp_out_dir),
            "--max-steps",
            str(int(args.max_steps)),
        ]
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            return rc

        best_candidates = []
        for p in sp_out_dir.glob("checkpoint_best_val_loss_*.speaker.safetensors"):
            m = best_re.match(p.name)
            if m:
                best_candidates.append((float(m.group(2)), int(m.group(1)), p))
        if best_candidates:
            best_candidates.sort(key=lambda item: (item[0], item[1]))
            src = best_candidates[0][2]
        else:
            src = sp_out_dir / "checkpoint_final.speaker.safetensors"
        if not src.is_file():
            raise FileNotFoundError(f"Expected speaker inversion output not found in {sp_out_dir}")

        dst = out_root / f"{sp}.speaker.safetensors"
        shutil.copy2(src, dst)
        print(f"[INVERT] Saved {dst} ({dst.stat().st_size} bytes) from {src.name}", flush=True)

    return 0


def cmd_bench_cpu(_: argparse.Namespace) -> int:
    ckpt = TRAIN_OUTPUT_DIR / "model_best.safetensors"
    embed = ROOT / "outputs/sc29_v1_speakers/fuyuko.speaker.safetensors"
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    if not embed.is_file():
        raise FileNotFoundError(f"Speaker embedding not found: {embed}")

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    code = "\n".join([
        "import sys, time, torch",
        f"sys.path.insert(0, {str(ROOT / 'Irodori-TTS')!r})",
        "from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest",
        "print(f'[BENCH-CPU] torch_threads={torch.get_num_threads()} ckpt={sys.argv[1]}', flush=True)",
        "t_load = time.perf_counter()",
        "rt = InferenceRuntime.from_key(RuntimeKey(",
        "    checkpoint=sys.argv[1],",
        "    model_device='cpu',",
        "    codec_repo='Aratako/Semantic-DACVAE-Japanese-32dim',",
        "    model_precision='fp32',",
        "    codec_device='cpu',",
        "    codec_precision='fp32',",
        "    compile_model=False,",
        "))",
        "print(f'[BENCH-CPU] load_sec={time.perf_counter() - t_load:.2f}s', flush=True)",
        "rt.synthesize(SamplingRequest(text='こんにちは。', ref_embed=sys.argv[2], seconds=1.5, num_steps=2, cfg_scale=0.0, seed=42), log_fn=None)",
        "text = 'プロデューサーさん、今日のお仕事もふゆにお任せくださいねっ！'",
        "caption = '猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。'",
        "modes = [",
        "    ('meanflow_4nfe_equiv', dict(num_steps=4, cfg_scale=0.0)),",
        "    ('rf_sway8_joint', dict(num_steps=8, t_schedule_mode='sway', sway_coeff=-1.0, cfg_guidance_mode='joint', cfg_scale=3.0)),",
        "    ('rf_40_indep', dict(num_steps=40, cfg_guidance_mode='independent', cfg_scale_text=3.0, cfg_scale_caption=3.0, cfg_scale_speaker=5.0)),",
        "]",
        "for name, kw in modes:",
        "    req = SamplingRequest(text=text, caption=caption, ref_embed=sys.argv[2], seed=42, **kw)",
        "    res = rt.synthesize(req, log_fn=None)",
        "    audio_sec = float(res.audio.shape[-1]) / float(res.sample_rate)",
        "    rtf = float(res.total_to_decode) / max(1e-6, audio_sec)",
        "    stages = ', '.join(f'{k}={v * 1000.0:.1f}ms' for k, v in res.stage_timings)",
        "    print(f'[BENCH-CPU] mode={name} audio_sec={audio_sec:.2f}s total={res.total_to_decode:.2f}s RTF={rtf:.2f}x | {stages}', flush=True)",
    ])
    return subprocess.run([str(VENV_PY), "-c", code, str(ckpt), str(embed)], env=env).returncode


def cmd_probe_meanflow(args: argparse.Namespace) -> int:
    if not MEANFLOW_4090_CONFIG.is_file():
        raise FileNotFoundError(f"MeanFlow config not found: {MEANFLOW_4090_CONFIG}")
    if not TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(f"Train manifest not found: {TRAIN_MANIFEST}")

    probe_dir = Path("/tmp/sc29_meanflow_probe")
    if probe_dir.exists():
        shutil.rmtree(probe_dir, ignore_errors=True)
    probe_dir.mkdir(parents=True, exist_ok=True)
    tok_link = probe_dir / "tokenizer"
    tok_link.symlink_to(TOKENIZER_SRC)

    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    cmd = [
        str(VENV_PY),
        str(ROOT / "Irodori-TTS/train.py"),
        "--config",
        str(MEANFLOW_4090_CONFIG),
        "--manifest",
        str(TRAIN_MANIFEST),
        "--output-dir",
        str(probe_dir),
        "--max-steps",
        str(int(args.steps)),
        "--log-every",
        "2",
        "--save-every",
        "999999",
        "--checkpoint-best-n",
        "0",
        "--valid-ratio",
        "0.0",
        "--valid-every",
        "0",
    ]
    try:
        return subprocess.run(cmd, env=env).returncode
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audited single-entrypoint runner for SC29 pipeline.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    sub.add_parser("check", help="Run read-only pre-flight checks")

    p_p1 = sub.add_parser("prepare-p1", help="Prepare pending Wave A-P1+P2 character batches")
    p_p1.add_argument("--workers", type=int, default=12)

    p_wh = sub.add_parser("whisper-p1", help="Pre-run local Whisper on prepared character batches and stash cache")
    p_wh.add_argument("--no-stash", action="store_true", help="Skip stashing whisper_text after transcription")
    p_wh.add_argument("--batch-size", type=int, default=16, help="Clips per GPU batch")
    p_wh.add_argument("--benchmark", action="store_true", help="Run read-only speed/accuracy benchmark")
    p_wh.add_argument("--force", action="store_true", help="Force re-transcribing all prepared batches and overwrite cached whisper file")

    p_lat = sub.add_parser("extract-latents", help="Extract shared DACVAE latents into shared_latents/")
    p_lat.add_argument("--variant", choices=["lite", "full"], default="full")
    p_lat.add_argument("--workers", type=int, default=12)
    p_lat.add_argument("--limit", type=int, default=None)

    p_vm = sub.add_parser("verify-manifest", help="Verify LatentTextDataset loads manifest")
    p_vm.add_argument("--variant", choices=["lite", "full"], default="full")

    p_trn = sub.add_parser("train", help="Run 29-character Full FT training")
    p_trn.add_argument("--variant", choices=["lite", "full"], default="full")
    p_trn.add_argument("--resume", default=None)

    p_cvt = sub.add_parser("convert-best", help="Convert best .pt checkpoint inside variant output dir to safetensors")
    p_cvt.add_argument("--variant", choices=["lite", "full"], default="full")
    p_cvt.add_argument("--input", required=True)
    p_cvt.add_argument("--output", default=None)

    p_ev = sub.add_parser("eval", help="Run multi-character and Fuyuko dual-persona evaluation")
    p_ev.add_argument("--base-checkpoint", default=None)
    p_ev.add_argument("--checkpoint", default=None)
    p_ev.add_argument("--extra-checkpoint", action="append", default=[])
    p_ev.add_argument("--speakers", default=None)
    p_ev.add_argument("--out-dir", default=None)
    p_ev.add_argument("--port", type=int, default=None)
    p_ev.add_argument("--serve-only", action="store_true")

    p_inv = sub.add_parser("invert-speaker", help="Train Speaker Inversion embeddings per character")
    p_inv.add_argument("--speakers", nargs="+", required=True)
    p_inv.add_argument("--checkpoint", default=str(TRAIN_OUTPUT_DIR / "model_best.safetensors"))
    p_inv.add_argument("--out-root", default=str(ROOT / "outputs/sc29_v1_speakers"))
    p_inv.add_argument("--max-steps", type=int, default=1500)

    sub.add_parser("bench-cpu", help="Benchmark CPU-only fp32 synthesis latency with --ref-embed")

    p_mf = sub.add_parser("probe-meanflow", help="Run a short MeanFlow distillation speed/VRAM probe on 4090")
    p_mf.add_argument("--steps", type=int, default=6)

    args = parser.parse_args()
    dispatch = {
        "check": cmd_check,
        "prepare-p1": cmd_prepare_p1,
        "whisper-p1": cmd_whisper_p1,
        "extract-latents": cmd_extract_latents,
        "verify-manifest": cmd_verify_manifest,
        "train": cmd_train,
        "convert-best": cmd_convert_best,
        "eval": cmd_eval,
        "invert-speaker": cmd_invert_speaker,
        "bench-cpu": cmd_bench_cpu,
        "probe-meanflow": cmd_probe_meanflow,
    }
    return dispatch[args.subcommand](args)


if __name__ == "__main__":
    sys.exit(main())
