#!/usr/bin/env python3
"""Minimal pack/unpack utility for running SC29 MeanFlow distillation on Colab."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/darwinwsl/code/irodori")
DEFAULT_OLD_PREFIX = "/home/darwinwsl/code/irodori/datasets/shared_latents"


def _copy_if_needed(src: Path, dst: Path, *, force: bool) -> None:
    if not src.is_file():
        raise FileNotFoundError(f"Missing required file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.is_file() and dst.stat().st_size == src.stat().st_size:
        print(f"[SKIP] {dst} already matches size ({dst.stat().st_size} bytes)")
        return
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    print(f"[COPY] {src} -> {dst} ({dst.stat().st_size / (1024 ** 2):.1f} MiB)")


def _make_tar_from_dir(src_dir: Path, dst_tar: Path, *, force: bool) -> None:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {src_dir}")
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst_tar.is_file() and dst_tar.stat().st_size > 0:
        print(f"[SKIP] {dst_tar} already exists ({dst_tar.stat().st_size / (1024 ** 2):.1f} MiB)")
        return
    tmp = dst_tar.with_name(dst_tar.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    subprocess.run(["tar", "-cf", str(tmp), "-C", str(src_dir), "."], check=True)
    os.replace(tmp, dst_tar)
    print(f"[TAR ] {src_dir} -> {dst_tar} ({dst_tar.stat().st_size / (1024 ** 2):.1f} MiB)")


def _extract_tar(src_tar: Path, dst_dir: Path, *, force: bool) -> None:
    if not src_tar.is_file():
        raise FileNotFoundError(f"Missing archive: {src_tar}")
    if dst_dir.is_dir() and any(dst_dir.iterdir()) and not force:
        print(f"[SKIP] {dst_dir} already non-empty")
        return
    if force and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xf", str(src_tar), "-C", str(dst_dir)], check=True)
    print(f"[UNTAR] {src_tar} -> {dst_dir}")


def _rewrite_manifest(src: Path, dst: Path, *, old_prefix: str, new_prefix: str) -> int:
    old_norm = old_prefix.rstrip("/")
    new_norm = new_prefix.rstrip("/")
    lines_out: list[str] = []
    checked = 0
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            lp = str(row.get("latent_path", ""))
            if lp.startswith(old_norm + "/"):
                lp = new_norm + lp[len(old_norm):]
            elif lp == old_norm:
                lp = new_norm
            row["latent_path"] = lp
            if checked < 5:
                if not Path(lp).is_file():
                    raise FileNotFoundError(
                        f"Rewritten latent_path does not exist: {lp} (from {src})"
                    )
                checked += 1
            lines_out.append(json.dumps(row, ensure_ascii=False) + "\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    tmp.write_text("".join(lines_out), encoding="utf-8")
    os.replace(tmp, dst)
    print(f"[MANIFEST] {src} -> {dst} (rows={len(lines_out)}, verified_first={checked})")
    return len(lines_out)


def cmd_pack(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _copy_if_needed(Path(args.teacher).expanduser().resolve(), out_dir / "model_s8000.safetensors", force=args.force)
    _make_tar_from_dir(Path(args.tokenizer_dir).expanduser().resolve(), out_dir / "tokenizer.tar", force=args.force)
    _make_tar_from_dir(Path(args.latents_dir).expanduser().resolve(), out_dir / "shared_latents.tar", force=args.force)

    manifests_src = Path(args.manifests_dir).expanduser().resolve()
    for name in ("train_manifest.jsonl", "valid_manifest.jsonl"):
        src_m = manifests_src / name
        if src_m.is_file():
            _copy_if_needed(src_m, out_dir / "manifests" / name, force=args.force)
        elif name == "train_manifest.jsonl":
            raise FileNotFoundError(f"Missing required manifest: {src_m}")

    speakers_dir = Path(args.speakers_dir).expanduser().resolve()
    speaker_files = sorted(p.name for p in speakers_dir.glob("*.speaker.safetensors") if p.is_file())
    if speaker_files:
        if (speakers_dir / "inversion_summary.json").is_file():
            speaker_files.append("inversion_summary.json")
        spk_tar = out_dir / "speakers_s8000.tar"
        if not args.force and spk_tar.is_file() and spk_tar.stat().st_size > 0:
            print(f"[SKIP] {spk_tar} already exists ({spk_tar.stat().st_size / (1024 ** 2):.1f} MiB)")
        else:
            tmp = spk_tar.with_name(spk_tar.name + ".tmp")
            if tmp.exists():
                tmp.unlink()
            subprocess.run(["tar", "-cf", str(tmp), "-C", str(speakers_dir), *speaker_files], check=True)
            os.replace(tmp, spk_tar)
            print(f"[TAR ] {len(speaker_files)} speaker items -> {spk_tar} ({spk_tar.stat().st_size / (1024 ** 2):.1f} MiB)")
    else:
        print(f"[WARN] No top-level *.speaker.safetensors found in {speakers_dir}; skipped speakers_s8000.tar")

    total_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    print(f"\n[DONE] Bundle ready at {out_dir} (total={total_bytes / (1024 ** 3):.2f} GiB)")
    print("Upload this folder contents to Google Drive: MyDrive/irodori_sc29/bundle/")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    teacher_dir = work_dir / "teacher"
    latents_dir = work_dir / "shared_latents"
    manifests_dir = work_dir / "manifests"
    speakers_dir = work_dir / "speakers_s8000"
    output_dir = work_dir / "outputs" / "sc29_v1_meanflow"
    for d in (teacher_dir, latents_dir, manifests_dir, speakers_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    _copy_if_needed(bundle_dir / "model_s8000.safetensors", teacher_dir / "model_s8000.safetensors", force=args.force)
    _extract_tar(bundle_dir / "tokenizer.tar", teacher_dir / "tokenizer", force=args.force)
    _extract_tar(bundle_dir / "shared_latents.tar", latents_dir, force=args.force)

    tok_link = output_dir / "tokenizer"
    if tok_link.is_symlink() or tok_link.exists():
        if tok_link.is_symlink():
            tok_link.unlink()
        elif tok_link.is_dir():
            shutil.rmtree(tok_link)
        else:
            tok_link.unlink()
    tok_link.symlink_to(teacher_dir / "tokenizer")

    spk_tar = bundle_dir / "speakers_s8000.tar"
    if spk_tar.is_file():
        _extract_tar(spk_tar, speakers_dir, force=args.force)

    src_train = bundle_dir / "manifests" / "train_manifest.jsonl"
    _rewrite_manifest(
        src_train,
        manifests_dir / "train_manifest_colab.jsonl",
        old_prefix=args.old_prefix,
        new_prefix=str(latents_dir),
    )
    src_valid = bundle_dir / "manifests" / "valid_manifest.jsonl"
    if src_valid.is_file():
        _rewrite_manifest(
            src_valid,
            manifests_dir / "valid_manifest_colab.jsonl",
            old_prefix=args.old_prefix,
            new_prefix=str(latents_dir),
        )

    print(
        "\n[READY]\n"
        f"  teacher_checkpoint : {teacher_dir / 'model_s8000.safetensors'}\n"
        f"  train_manifest     : {manifests_dir / 'train_manifest_colab.jsonl'}\n"
        f"  output_dir         : {output_dir}\n"
        f"  speakers_s8000     : {speakers_dir}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Pack/unpack SC29 assets for Colab MeanFlow distillation.")
    sub = parser.add_subparsers(dest="subcommand", required=True)

    p_pack = sub.add_parser("pack", help="Pack local SC29 assets into a Drive-uploadable bundle")
    p_pack.add_argument("--out-dir", default=str(ROOT / "outputs/colab_meanflow_bundle"))
    p_pack.add_argument("--teacher", default=str(ROOT / "outputs/sc29_v1_full/model_s8000.safetensors"))
    p_pack.add_argument("--tokenizer-dir", default=str(ROOT / "models/Irodori-TTS-v4.1-Small/tokenizer"))
    p_pack.add_argument("--latents-dir", default=str(ROOT / "datasets/shared_latents"))
    p_pack.add_argument("--manifests-dir", default=str(ROOT / "datasets/sc29_v1"))
    p_pack.add_argument("--speakers-dir", default=str(ROOT / "outputs/sc29_v1_speakers_s8000"))
    p_pack.add_argument("--force", action="store_true", help="Overwrite existing files/archives")

    p_unpack = sub.add_parser("unpack", help="Unpack bundle in Colab and rewrite manifest latent paths")
    p_unpack.add_argument("--bundle-dir", default="/content/drive/MyDrive/irodori_sc29/bundle")
    p_unpack.add_argument("--work-dir", default="/content/sc29_work")
    p_unpack.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX)
    p_unpack.add_argument("--force", action="store_true", help="Re-extract even if target directories exist")

    args = parser.parse_args()
    if args.subcommand == "pack":
        return cmd_pack(args)
    return cmd_unpack(args)


if __name__ == "__main__":
    sys.exit(main())
