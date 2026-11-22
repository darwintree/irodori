#!/usr/bin/env python3
"""Minimal pack/unpack utility with 500 MiB split parts & SHA256 verification for Colab MeanFlow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional

ROOT = Path("/home/darwinwsl/code/irodori")
DEFAULT_OLD_PREFIX = "/home/darwinwsl/code/irodori/datasets/shared_latents"
CHUNK_SIZE = 4 * 1024 * 1024
PART_SUFFIX_RE = re.compile(r"^(.+)\.part(\d+)$")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(CHUNK_SIZE)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _remove_base_and_parts(out_dir: Path, rel_name: str) -> None:
    base = out_dir / rel_name
    if base.exists() or base.is_symlink():
        base.unlink()
    parent = base.parent
    if parent.is_dir():
        prefix = base.name + ".part"
        for p in parent.iterdir():
            if p.name.startswith(prefix) and (p.is_file() or p.is_symlink()):
                p.unlink()


def _resolve_sources(bundle_dir: Path, rel_name: str) -> List[Path]:
    single = bundle_dir / rel_name
    if single.is_file():
        return [single]

    parent = single.parent
    indexed: List[tuple] = []
    if parent.is_dir():
        for p in parent.iterdir():
            if not p.is_file():
                continue
            m = PART_SUFFIX_RE.match(p.name)
            if m and m.group(1) == single.name:
                indexed.append((int(m.group(2)), p))

    if not indexed:
        raise FileNotFoundError(
            f"Missing bundle file or split parts for {rel_name} under {bundle_dir}"
        )

    indexed.sort(key=lambda item: item[0])
    expected_indices = list(range(len(indexed)))
    actual_indices = [idx for idx, _ in indexed]
    if actual_indices != expected_indices:
        raise FileNotFoundError(
            f"Non-contiguous split parts for {rel_name}: found indices {actual_indices}, "
            f"expected {expected_indices}"
        )
    return [p for _, p in indexed]


def _stream_to_split_files(
    stream: BinaryIO,
    out_dir: Path,
    rel_name: str,
    part_bytes: int,
) -> Dict[str, str]:
    base = out_dir / rel_name
    base.parent.mkdir(parents=True, exist_ok=True)
    hashes: Dict[str, str] = {}
    part_idx = 0
    written_in_part = 0
    out_f: Optional[BinaryIO] = None
    tmp_path: Optional[Path] = None
    final_path: Optional[Path] = None
    part_hasher = hashlib.sha256()

    def _open_next_part() -> None:
        nonlocal out_f, tmp_path, final_path, part_hasher, written_in_part
        rel_part = f"{rel_name}.part{part_idx:03d}"
        final_path = out_dir / rel_part
        tmp_path = final_path.with_name(final_path.name + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        out_f = tmp_path.open("wb")
        part_hasher = hashlib.sha256()
        written_in_part = 0

    def _close_current_part() -> None:
        nonlocal out_f, tmp_path, final_path, part_idx
        if out_f is None or tmp_path is None or final_path is None:
            return
        out_f.close()
        out_f = None
        os.replace(tmp_path, final_path)
        rel_part = final_path.relative_to(out_dir).as_posix()
        hashes[rel_part] = part_hasher.hexdigest()
        size_mib = final_path.stat().st_size / (1024 ** 2)
        print(f"[PART] {rel_part} ({size_mib:.1f} MiB)", flush=True)
        tmp_path = None
        final_path = None
        part_idx += 1

    try:
        while True:
            if out_f is None:
                _open_next_part()
            remaining = part_bytes - written_in_part
            chunk = stream.read(min(CHUNK_SIZE, remaining))
            if not chunk:
                break
            assert out_f is not None
            out_f.write(chunk)
            part_hasher.update(chunk)
            written_in_part += len(chunk)
            if written_in_part >= part_bytes:
                _close_current_part()
        if out_f is not None:
            if written_in_part > 0 or part_idx == 0:
                _close_current_part()
            else:
                out_f.close()
                out_f = None
                if tmp_path is not None and tmp_path.exists():
                    tmp_path.unlink()
    except Exception:
        if out_f is not None:
            out_f.close()
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
        _remove_base_and_parts(out_dir, rel_name)
        raise

    return hashes


def _pack_file(
    src: Path,
    out_dir: Path,
    rel_name: str,
    *,
    split_bytes: int,
    force: bool,
) -> Dict[str, str]:
    if not src.is_file():
        raise FileNotFoundError(f"Missing required file: {src}")

    if split_bytes > 0:
        unsplit = out_dir / rel_name
        if unsplit.exists() or unsplit.is_symlink():
            unsplit.unlink()
        if not force:
            try:
                existing = _resolve_sources(out_dir, rel_name)
                src_size = src.stat().st_size
                sizes = [p.stat().st_size for p in existing]
                if (
                    existing[0].name != Path(rel_name).name
                    and sum(sizes) == src_size
                    and all(s == split_bytes for s in sizes[:-1])
                    and (sizes[-1] <= split_bytes)
                ):
                    print(f"[SKIP] {rel_name} ({len(existing)} parts match {src_size} bytes)")
                    return {
                        p.relative_to(out_dir).as_posix(): _sha256_file(p)
                        for p in existing
                    }
            except FileNotFoundError:
                pass
        _remove_base_and_parts(out_dir, rel_name)
        print(f"[SPLIT] {src} -> {rel_name}.part* ({split_bytes // (1024 ** 2)} MiB/part)", flush=True)
        with src.open("rb") as f:
            return _stream_to_split_files(f, out_dir, rel_name, split_bytes)

    _remove_base_and_parts(out_dir, rel_name)
    dst = out_dir / rel_name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.is_file() and dst.stat().st_size == src.stat().st_size:
        print(f"[SKIP] {rel_name} already matches size ({dst.stat().st_size} bytes)")
    else:
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        print(f"[COPY] {src} -> {rel_name} ({dst.stat().st_size / (1024 ** 2):.1f} MiB)")
    return {rel_name: _sha256_file(dst)}


def _pack_tar(
    src_dir: Path,
    out_dir: Path,
    rel_name: str,
    *,
    split_bytes: int,
    force: bool,
    members: Optional[List[str]] = None,
) -> Dict[str, str]:
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {src_dir}")

    tar_targets = members if members is not None else ["."]
    if split_bytes > 0:
        unsplit = out_dir / rel_name
        if unsplit.exists() or unsplit.is_symlink():
            unsplit.unlink()
        if not force:
            try:
                existing = _resolve_sources(out_dir, rel_name)
                if existing[0].name != Path(rel_name).name and all(p.stat().st_size > 0 for p in existing):
                    print(f"[SKIP] {rel_name} ({len(existing)} parts already exist)")
                    return {
                        p.relative_to(out_dir).as_posix(): _sha256_file(p)
                        for p in existing
                    }
            except FileNotFoundError:
                pass
        _remove_base_and_parts(out_dir, rel_name)
        print(f"[TAR+SPLIT] {src_dir} -> {rel_name}.part* ({split_bytes // (1024 ** 2)} MiB/part)", flush=True)
        proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(src_dir), *tar_targets],
            stdout=subprocess.PIPE,
        )
        assert proc.stdout is not None
        try:
            hashes = _stream_to_split_files(proc.stdout, out_dir, rel_name, split_bytes)
        except Exception:
            proc.kill()
            proc.wait()
            raise
        rc = proc.wait()
        if rc != 0:
            _remove_base_and_parts(out_dir, rel_name)
            raise subprocess.CalledProcessError(rc, proc.args)
        return hashes

    _remove_base_and_parts(out_dir, rel_name)
    dst_tar = out_dir / rel_name
    dst_tar.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst_tar.is_file() and dst_tar.stat().st_size > 0:
        print(f"[SKIP] {rel_name} already exists ({dst_tar.stat().st_size / (1024 ** 2):.1f} MiB)")
    else:
        tmp = dst_tar.with_name(dst_tar.name + ".tmp")
        if tmp.exists():
            tmp.unlink()
        subprocess.run(["tar", "-cf", str(tmp), "-C", str(src_dir), *tar_targets], check=True)
        os.replace(tmp, dst_tar)
        print(f"[TAR ] {src_dir} -> {rel_name} ({dst_tar.stat().st_size / (1024 ** 2):.1f} MiB)")
    return {rel_name: _sha256_file(dst_tar)}


def _load_and_precheck_sha256sums(bundle_dir: Path) -> Optional[Dict[str, str]]:
    sums_path = bundle_dir / "SHA256SUMS"
    if not sums_path.is_file():
        return None
    expected: Dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"Malformed SHA256SUMS line: {line!r}")
        digest, rel = parts[0].lower(), parts[1].lstrip("*")
        expected[rel] = digest

    missing = [rel for rel in sorted(expected) if not (bundle_dir / rel).is_file()]
    if missing:
        raise FileNotFoundError(
            "Bundle is incomplete (files listed in SHA256SUMS are missing):\n  - "
            + "\n  - ".join(missing)
        )
    print(f"[CHECK] Found all {len(expected)} files listed in SHA256SUMS", flush=True)
    return expected


def _iter_verified_chunks(
    sources: List[Path],
    bundle_dir: Path,
    expected_sums: Optional[Dict[str, str]],
) -> Iterator[bytes]:
    for src in sources:
        rel = src.relative_to(bundle_dir).as_posix()
        h = hashlib.sha256()
        with src.open("rb") as f:
            while True:
                buf = f.read(CHUNK_SIZE)
                if not buf:
                    break
                h.update(buf)
                yield buf
        actual = h.hexdigest()
        if expected_sums is not None and rel in expected_sums:
            if actual != expected_sums[rel]:
                raise ValueError(
                    f"SHA256 mismatch for {rel}: expected {expected_sums[rel]}, got {actual}"
                )
        print(f"[OK  ] Verified {rel} ({src.stat().st_size / (1024 ** 2):.1f} MiB)", flush=True)


def _materialize_file(
    bundle_dir: Path,
    rel_name: str,
    dst: Path,
    *,
    expected_sums: Optional[Dict[str, str]],
    force: bool,
) -> None:
    sources = _resolve_sources(bundle_dir, rel_name)
    total_size = sum(p.stat().st_size for p in sources)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not force and dst.is_file() and dst.stat().st_size == total_size:
        print(f"[SKIP] {dst} already matches size ({total_size} bytes)")
        return

    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with tmp.open("wb") as out_f:
            for chunk in _iter_verified_chunks(sources, bundle_dir, expected_sums):
                out_f.write(chunk)
        os.replace(tmp, dst)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    print(f"[FILE] {rel_name} ({len(sources)} part(s)) -> {dst} ({dst.stat().st_size / (1024 ** 2):.1f} MiB)")


def _extract_tar_stream(
    bundle_dir: Path,
    rel_name: str,
    dst_dir: Path,
    *,
    expected_sums: Optional[Dict[str, str]],
    force: bool,
) -> None:
    sources = _resolve_sources(bundle_dir, rel_name)
    if dst_dir.is_dir() and any(dst_dir.iterdir()) and not force:
        print(f"[SKIP] {dst_dir} already non-empty")
        return
    if force and dst_dir.is_dir():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        ["tar", "-xf", "-", "-C", str(dst_dir)],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    broken_pipe = False
    try:
        for chunk in _iter_verified_chunks(sources, bundle_dir, expected_sums):
            if not broken_pipe:
                try:
                    proc.stdin.write(chunk)
                except BrokenPipeError:
                    broken_pipe = True
        if not broken_pipe:
            proc.stdin.close()
        rc = proc.wait()
    except Exception:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise

    if rc != 0:
        raise subprocess.CalledProcessError(rc, proc.args)
    print(f"[UNTAR] {rel_name} ({len(sources)} part(s)) -> {dst_dir}")


def _verify_single_file_sha(
    path: Path,
    bundle_dir: Path,
    expected_sums: Optional[Dict[str, str]],
) -> None:
    if expected_sums is None:
        return
    rel = path.relative_to(bundle_dir).as_posix()
    if rel in expected_sums:
        actual = _sha256_file(path)
        if actual != expected_sums[rel]:
            raise ValueError(f"SHA256 mismatch for {rel}: expected {expected_sums[rel]}, got {actual}")
        print(f"[OK  ] Verified {rel}", flush=True)


def _rewrite_manifest(
    src: Path,
    dst: Path,
    *,
    bundle_dir: Path,
    expected_sums: Optional[Dict[str, str]],
    old_prefix: str,
    new_prefix: str,
) -> int:
    _verify_single_file_sha(src, bundle_dir, expected_sums)
    old_norm = old_prefix.rstrip("/")
    new_norm = new_prefix.rstrip("/")
    lines_out: List[str] = []
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
    if args.split_mb < 0:
        raise ValueError(f"--split-mb must be >= 0, got {args.split_mb}")
    split_bytes = int(args.split_mb) * 1024 * 1024

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_hashes: Dict[str, str] = {}
    all_hashes.update(
        _pack_file(
            Path(args.teacher).expanduser().resolve(),
            out_dir,
            "model_s8000.safetensors",
            split_bytes=split_bytes,
            force=args.force,
        )
    )
    all_hashes.update(
        _pack_tar(
            Path(args.tokenizer_dir).expanduser().resolve(),
            out_dir,
            "tokenizer.tar",
            split_bytes=0,
            force=args.force,
        )
    )
    all_hashes.update(
        _pack_tar(
            Path(args.latents_dir).expanduser().resolve(),
            out_dir,
            "shared_latents.tar",
            split_bytes=split_bytes,
            force=args.force,
        )
    )

    manifests_src = Path(args.manifests_dir).expanduser().resolve()
    for name in ("train_manifest.jsonl", "valid_manifest.jsonl"):
        src_m = manifests_src / name
        if src_m.is_file():
            all_hashes.update(
                _pack_file(
                    src_m,
                    out_dir,
                    f"manifests/{name}",
                    split_bytes=0,
                    force=args.force,
                )
            )
        elif name == "train_manifest.jsonl":
            raise FileNotFoundError(f"Missing required manifest: {src_m}")

    speakers_dir = Path(args.speakers_dir).expanduser().resolve()
    speaker_files = sorted(p.name for p in speakers_dir.glob("*.speaker.safetensors") if p.is_file())
    if speaker_files:
        if (speakers_dir / "inversion_summary.json").is_file():
            speaker_files.append("inversion_summary.json")
        all_hashes.update(
            _pack_tar(
                speakers_dir,
                out_dir,
                "speakers_s8000.tar",
                split_bytes=0,
                force=args.force,
                members=speaker_files,
            )
        )
    else:
        print(f"[WARN] No top-level *.speaker.safetensors found in {speakers_dir}; skipped speakers_s8000.tar")

    sums_lines = [f"{all_hashes[rel]}  {rel}\n" for rel in sorted(all_hashes)]
    sums_path = out_dir / "SHA256SUMS"
    sums_tmp = sums_path.with_name("SHA256SUMS.tmp")
    sums_tmp.write_text("".join(sums_lines), encoding="utf-8")
    os.replace(sums_tmp, sums_path)

    total_bytes = sum((out_dir / rel).stat().st_size for rel in all_hashes) + sums_path.stat().st_size
    print(f"\n[DONE] Bundle ready at {out_dir} ({len(all_hashes)} files + SHA256SUMS, total={total_bytes / (1024 ** 3):.2f} GiB)")
    print("Upload ALL files/folders in this directory to Google Drive: MyDrive/irodori_sc29/bundle/")
    return 0


def cmd_unpack(args: argparse.Namespace) -> int:
    bundle_dir = Path(args.bundle_dir).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir}")

    expected_sums = None if args.skip_verify else _load_and_precheck_sha256sums(bundle_dir)

    teacher_dir = work_dir / "teacher"
    latents_dir = work_dir / "shared_latents"
    manifests_dir = work_dir / "manifests"
    speakers_dir = work_dir / "speakers_s8000"
    output_dir = work_dir / "outputs" / "sc29_v1_meanflow"
    for d in (teacher_dir, latents_dir, manifests_dir, speakers_dir, output_dir):
        d.mkdir(parents=True, exist_ok=True)

    _materialize_file(
        bundle_dir,
        "model_s8000.safetensors",
        teacher_dir / "model_s8000.safetensors",
        expected_sums=expected_sums,
        force=args.force,
    )
    _extract_tar_stream(
        bundle_dir,
        "tokenizer.tar",
        teacher_dir / "tokenizer",
        expected_sums=expected_sums,
        force=args.force,
    )
    _extract_tar_stream(
        bundle_dir,
        "shared_latents.tar",
        latents_dir,
        expected_sums=expected_sums,
        force=args.force,
    )

    tok_link = output_dir / "tokenizer"
    if tok_link.is_symlink() or tok_link.exists():
        if tok_link.is_symlink():
            tok_link.unlink()
        elif tok_link.is_dir():
            shutil.rmtree(tok_link)
        else:
            tok_link.unlink()
    tok_link.symlink_to(teacher_dir / "tokenizer")

    try:
        _resolve_sources(bundle_dir, "speakers_s8000.tar")
        has_speakers = True
    except FileNotFoundError:
        has_speakers = False
    if has_speakers:
        _extract_tar_stream(
            bundle_dir,
            "speakers_s8000.tar",
            speakers_dir,
            expected_sums=expected_sums,
            force=args.force,
        )

    src_train = bundle_dir / "manifests" / "train_manifest.jsonl"
    _rewrite_manifest(
        src_train,
        manifests_dir / "train_manifest_colab.jsonl",
        bundle_dir=bundle_dir,
        expected_sums=expected_sums,
        old_prefix=args.old_prefix,
        new_prefix=str(latents_dir),
    )
    src_valid = bundle_dir / "manifests" / "valid_manifest.jsonl"
    if src_valid.is_file():
        _rewrite_manifest(
            src_valid,
            manifests_dir / "valid_manifest_colab.jsonl",
            bundle_dir=bundle_dir,
            expected_sums=expected_sums,
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
    p_pack.add_argument("--split-mb", type=int, default=500, help="Split large files into N MiB parts (0 to disable)")
    p_pack.add_argument("--force", action="store_true", help="Overwrite existing files/archives")

    p_unpack = sub.add_parser("unpack", help="Unpack bundle in Colab and rewrite manifest latent paths")
    p_unpack.add_argument("--bundle-dir", default="/content/drive/MyDrive/irodori_sc29/bundle")
    p_unpack.add_argument("--work-dir", default="/content/sc29_work")
    p_unpack.add_argument("--old-prefix", default=DEFAULT_OLD_PREFIX)
    p_unpack.add_argument("--skip-verify", action="store_true", help="Skip SHA256SUMS verification")
    p_unpack.add_argument("--force", action="store_true", help="Re-extract even if target directories exist")

    args = parser.parse_args()
    if args.subcommand == "pack":
        return cmd_pack(args)
    return cmd_unpack(args)


if __name__ == "__main__":
    sys.exit(main())
