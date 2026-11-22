#!/usr/bin/env python3
"""Unified multi-checkpoint evaluation for SC29 (Base vs v1_lite vs v2_full).

Evaluates held-out leak-free speaker references across:
  - Section A: Fuyuko dual-persona switch matrix (3 texts x 3 triggers = 9 clips/model)
  - Section B: 8-character multi-emotion contrasts (8 speakers x 3 emotions = 24 clips/model)

Computes acoustic metrics (duration, RMS dB, median F0 Hz) + local batched Whisper large-v3 CER,
and writes summary.json + self-contained index.html (+ optional HTTP serving via --port).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import gc
import html
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import wave

ROOT = Path("/home/darwinwsl/code/irodori")
DEFAULT_BASE_CKPT = ROOT / "models/Irodori-TTS-v4.1-Small/model.safetensors"
DEFAULT_VALID_CANDIDATES = [
    ROOT / "datasets/sc29_v1/valid_manifest.jsonl",
    ROOT / "datasets/sc29_v1_lite/valid_manifest.jsonl",
]
DEFAULT_TRAIN_EXCLUDES = [
    ROOT / "datasets/sc29_v1/train_manifest.jsonl",
    ROOT / "datasets/sc29_v1_lite/train_manifest.jsonl",
]
DEFAULT_OUT_DIR = ROOT / "outputs/sc29_v1_eval"
DEFAULT_WHISPER_PY = Path("/home/darwinwsl/miniconda3/envs/whisperx/bin/python")
IRODORI_REPO = ROOT / "Irodori-TTS"
VCA_REPO = ROOT / "voice-caption-annotator"

EVAL_SPEAKERS: List[str] = [
    "mano",
    "fuyuko",
    "madoka",
    "luca",
    "hana",
    "hazuki",
    "kaho",
    "sakuya",
]

SPEAKER_NAMES_JA: Dict[str, str] = {
    "mano": "櫻木真乃",
    "fuyuko": "黛冬優子",
    "madoka": "樋口円香",
    "luca": "斑鳩ルカ",
    "hana": "鈴木羽那",
    "hazuki": "七草はづき",
    "kaho": "小宮果穂",
    "sakuya": "白瀬咲耶",
}

FUYUKO_PERSONA_TRIGGERS: List[Tuple[str, str, str]] = [
    (
        "omote",
        "表 (omote / 営業)",
        "猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。",
    ),
    (
        "ura",
        "裏 (ura / 素の本音)",
        "素の本音口調で、低めの少しぶっきらぼうなトーンで呆れたように話している。",
    ),
    (
        "switch",
        "切替 (switch / 漏れ出し)",
        "営業口調から素が漏れ出しながら、前半は可愛らしく話し後半で思わず低い地声が出ている。",
    ),
]

FUYUKO_MATRIX_TEXTS: List[Tuple[str, str]] = [
    ("t1_work", "プロデューサーさん、今日のお仕事もふゆにお任せくださいねっ！"),
    ("t2_listen", "もう……あんた、ちゃんとふゆの話聞いてたんですか？"),
    ("t3_gaze", "えへへ、ありがとうございます！……って、そんなに見つめられたらやりにくいんですけど。"),
]

# Section B: 8 speakers x 3 contrasting emotions (calm / bright_joy / intense_emo)
SECTION_B_SPECS: List[Tuple[str, str, str, str, str]] = [
    # mano
    ("mano", "calm", "穏やか", "穏やかで優しい口調で、相手を気遣いながら静かに話しかけている。", "……あの、今、少しだけお時間よろしいですか？"),
    ("mano", "bright_joy", "明るい歓喜", "明るく弾んだ口調で、心から嬉しそうに元気よく笑いかけている。", "プロデューサーさん！今日はとってもいいお天気ですねっ！"),
    ("mano", "intense_emo", "泣き・切迫", "不安で声を小さく震わせながら、今にも泣き出しそうな弱い声で訴えている。", "ごめんなさい……私のせいで、みんなに迷惑をかけてしまって……"),
    # fuyuko
    ("fuyuko", "calm", "丁寧営業", "猫かぶりな営業口調で、丁寧で落ち着いた優しい声で礼儀正しく挨拶している。", "お疲れ様です！今日も一日、よろしくお願いしますね。"),
    ("fuyuko", "bright_joy", "甘え営業", "猫かぶりな営業口調で、甘く可愛らしく弾んだ高い声で嬉しそうに話している。", "えへへ、ふゆのステージ、楽しみにしててくださいねっ！"),
    ("fuyuko", "intense_emo", "本音激昂", "素の本音口調で、苛立ちを抑えきれず強く鋭い声で言い放っている。", "ちょっとあんた、本気でそんなこと言ってんの！？信じらんないっ！"),
    # madoka
    ("madoka", "calm", "冷淡", "感情を抑えた冷たく淡々とした低い口調で、素っ気なく短く返事をしている。", "……別に。どうぞ、お構いなく。"),
    ("madoka", "bright_joy", "皮肉な微笑", "少し皮肉混じりに口角を上げながら、静かで落ち着いた微笑みを含んで話している。", "ふっ……相変わらずおめでたいですね、ミスター・好青年。"),
    ("madoka", "intense_emo", "強張り・覚悟", "強張った声に本気の悔しさと熱意を滲ませながら、きっぱりと言い切っている。", "私、あなたには期待しません。代わりに行動で……信じさせてください。"),
    # luca
    ("luca", "calm", "投げやり低音", "低く乾いた落ち着いた声で、面倒くさそうに間を取りながらぼそっと呟いている。", "……知るかよ。勝手にやってろ。"),
    ("luca", "bright_joy", "不器用な感謝", "肩の力を抜いた少し柔らかな声で、控えめながら素直に感謝を伝えている。", "……あんたさ、今日だけは、ちょっとありがとよ。"),
    ("luca", "intense_emo", "激しい怒号", "激しい怒りと苦しさを滲ませて声を張り上げ、突き放すように鋭く叫んでいる。", "うるせェっ！私に構うなって言ってるだろ……！"),
    # hana
    ("hana", "calm", "穏やか包容", "おっとりとした柔らかく穏やかな口調で、相手を安心させるように優しく語っている。", "うんっ、大丈夫だよ。あたしも一緒にいるから、心配しないで？"),
    ("hana", "bright_joy", "無邪気笑い", "屈託なく明るく元気いっぱいに笑い声を交えながら、楽しそうに弾んで話している。", "あはっ！これ、すっごく可愛いーっ！プロデューサーも見て見てっ！"),
    ("hana", "intense_emo", "震える本心", "寂しさと不安を抱えながら少し声を震わせ、必死に想いをまっすぐ伝えている。", "あたしのこと、もっと知ってほしいの……守られたいだけじゃないんだよ。"),
    # hazuki
    ("hazuki", "calm", "おっとり労い", "おっとりとして優しく落ち着いたお姉さんのような口調で、ゆったりと労わっている。", "プロデューサーさん、お疲れ様です。温かいお茶、淹れておきましたよ。"),
    ("hazuki", "bright_joy", "柔らかな笑み", "嬉しそうにふふっと柔らかな笑みを浮かべ、少しおどけた明るい声で話している。", "ふふっ、大成功ですね。今日のおやつは奮発しちゃいました！"),
    ("hazuki", "intense_emo", "凛とした忠告", "普段の穏やかさから一転し、凛とした真剣で張り詰めた声で注意を促している。", "いけませんよ、無理は禁物です。少し休憩を取りましょう？"),
    # kaho
    ("kaho", "calm", "丁寧報告", "少し背伸びして丁寧にはきはきと、真面目で素直な可愛い声で報告している。", "プロデューサーさん！今日のスケジュール、しっかり確認してきましたっ！"),
    ("kaho", "bright_joy", "全力興奮", "目を輝かせるように力強く弾んだ高い声で、興奮して元気いっぱいに叫んでいる。", "スゴいですっ！まるであたしの好きなヒーローみたいですーっ！"),
    ("kaho", "intense_emo", "悔し泣き我慢", "涙を堪えて声を詰まらせながら、悔しさと必死さを滲ませて訴えている。", "うっ……負けるもんかっ！あたし、絶対にあきらめませんから……！"),
    # sakuya
    ("sakuya", "calm", "優雅な気遣い", "涼やかで凛とした中性的で落ち着いた声で、優雅に優しくエスコートしている。", "おや、浮かない顔だね。よろしければ、私に話を聞かせてくれるかな。"),
    ("sakuya", "bright_joy", "朗らか誇り", "晴れやかで堂々とした誇らしげな声で、楽しそうに朗らかに笑いかけている。", "フフッ、美しいユニゾンだ！さあ、最高のステージにしようじゃないか！"),
    ("sakuya", "intense_emo", "切実な孤独", "孤独と切実な想いを抑え込んだ低く震える声で、静かに強く訴えている。", "お願いだ……誰か一人でもいい、私を置いていかないでくれ……。"),
]

_CER_STRIP_RE = re.compile(r"[\s。、！？…・ー「」『』（）()!?.,〜~—―]+")


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    section: str  # "A" or "B"
    speaker_id: str
    category: str  # trigger id or emotion key
    category_label: str
    text: str
    caption: str
    text_group: str = ""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    checkpoint: Path
    speaker_mode: str = "ref"  # "ref", "embed", or "noref"
    embed_dir: Optional[Path] = None


def build_prompt_suite() -> List[EvalPrompt]:
    prompts: List[EvalPrompt] = []
    for text_id, text in FUYUKO_MATRIX_TEXTS:
        for trig_id, trig_label, caption in FUYUKO_PERSONA_TRIGGERS:
            prompts.append(
                EvalPrompt(
                    prompt_id=f"A_fuyuko_{text_id}_{trig_id}",
                    section="A",
                    speaker_id="fuyuko",
                    category=trig_id,
                    category_label=trig_label,
                    text=text,
                    caption=caption,
                    text_group=text_id,
                )
            )
    for sp, emo_key, emo_label, caption, text in SECTION_B_SPECS:
        prompts.append(
            EvalPrompt(
                prompt_id=f"B_{sp}_{emo_key}",
                section="B",
                speaker_id=sp,
                category=emo_key,
                category_label=emo_label,
                text=text,
                caption=caption,
                text_group=sp,
            )
        )
    return prompts


def _parse_ckpt_spec(
    raw: str, default_label: Optional[str] = None
) -> Tuple[Optional[str], Path, str, Optional[Path]]:
    label = default_label
    rhs = raw
    if "=" in raw:
        lhs, rhs_part = raw.split("=", 1)
        if lhs.strip() and rhs_part.strip():
            label = lhs.strip()
            rhs = rhs_part.strip()

    speaker_mode = "ref"
    embed_dir: Optional[Path] = None
    path_str = rhs.strip()
    if "@embed:" in path_str:
        path_str, embed_dir_str = path_str.split("@embed:", 1)
        speaker_mode = "embed"
        embed_dir = Path(embed_dir_str.strip()).expanduser().resolve()
    elif path_str.endswith("@noref"):
        path_str = path_str[: -len("@noref")]
        speaker_mode = "noref"
    elif path_str.endswith("@ref"):
        path_str = path_str[: -len("@ref")]
        speaker_mode = "ref"

    return label, Path(path_str.strip()).expanduser(), speaker_mode, embed_dir


def _derive_model_id_and_name(label_hint: Optional[str], ckpt: Path, used_ids: set) -> Tuple[str, str]:
    parent_name = ckpt.parent.name
    if label_hint:
        base_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", label_hint).strip("_") or "model"
        disp = label_hint
    elif "sc29_v1_lite" in parent_name:
        base_id, disp = "v1_lite", "v1_lite (Wave A-P0 + Wave B)"
    elif "sc29_v1_full" in parent_name:
        base_id, disp = "v2_full", "v2_full (Wave A-P0+P1+P2 + Wave B)"
    elif "Irodori-TTS" in parent_name:
        base_id, disp = "base", "Base Zero-Shot (v4.1-Small)"
    else:
        base_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", parent_name).strip("_") or "ckpt"
        disp = f"{parent_name}/{ckpt.name}"

    cand = base_id
    idx = 2
    while cand in used_ids:
        cand = f"{base_id}_{idx}"
        idx += 1
    used_ids.add(cand)
    return cand, disp


def resolve_model_specs(args: argparse.Namespace) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    seen_keys: set = set()
    used_ids: set = set()

    def _add(
        label: Optional[str],
        path: Path,
        speaker_mode: str = "ref",
        embed_dir: Optional[Path] = None,
        fallback_disp: Optional[str] = None,
    ) -> None:
        resolved = path.resolve()
        dedup_key = (resolved, speaker_mode, embed_dir)
        if dedup_key in seen_keys:
            return
        if speaker_mode == "embed" and (embed_dir is None or not embed_dir.is_dir()):
            raise FileNotFoundError(f"Speaker embedding directory not found: {embed_dir}")
        seen_keys.add(dedup_key)
        mid, disp = _derive_model_id_and_name(label, path, used_ids)
        if fallback_disp and not label:
            disp = fallback_disp
        if speaker_mode != "ref" and not label:
            disp = f"{disp} [{speaker_mode}]"
        specs.append(
            ModelSpec(
                model_id=mid,
                display_name=disp,
                checkpoint=resolved,
                speaker_mode=speaker_mode,
                embed_dir=embed_dir,
            )
        )

    if args.base_checkpoint and str(args.base_checkpoint).lower() not in {"none", "off"}:
        lbl, base_p, sp_mode, emb_dir = _parse_ckpt_spec(str(args.base_checkpoint), default_label="base")
        if not base_p.is_file():
            raise FileNotFoundError(f"Base checkpoint not found: {base_p}")
        _add(lbl, base_p, sp_mode, emb_dir, fallback_disp="Base Zero-Shot (v4.1-Small)")

    for raw_extra in args.extra_checkpoint:
        lbl, p, sp_mode, emb_dir = _parse_ckpt_spec(raw_extra)
        if not p.is_file():
            raise FileNotFoundError(f"Extra checkpoint not found: {p}")
        _add(lbl, p, sp_mode, emb_dir)

    if args.checkpoint:
        lbl, p, sp_mode, emb_dir = _parse_ckpt_spec(args.checkpoint)
        if not p.is_file():
            raise FileNotFoundError(f"Primary checkpoint not found: {p}")
        _add(lbl, p, sp_mode, emb_dir)

    if not specs:
        raise ValueError("No valid checkpoints configured for evaluation.")
    return specs


def resolve_valid_manifest(raw_path: Optional[Path]) -> Path:
    if raw_path is not None:
        p = raw_path.expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"Requested --valid-manifest not found: {p}")
    for cand in DEFAULT_VALID_CANDIDATES:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        "No valid manifest found in default locations: " + ", ".join(str(p) for p in DEFAULT_VALID_CANDIDATES)
    )


def _pick_clips_for_speaker(
    pool: Sequence[Dict[str, Any]],
    max_clips: int,
    min_clips: int,
    target_total_frames: int,
) -> List[Dict[str, Any]]:
    preferred = [r for r in pool if 100 <= int(r.get("num_frames") or 0) <= 250]
    others = [r for r in pool if not (100 <= int(r.get("num_frames") or 0) <= 250)]

    def _sort_key(r: Dict[str, Any]) -> Tuple[int, int, str]:
        has_cap = 1 if (r.get("caption") or "").strip() else 0
        nf = int(r.get("num_frames") or 0)
        dist = abs(nf - 150)
        return (has_cap, dist, str(r.get("voice_key") or r.get("latent_path") or ""))

    ordered = sorted(preferred, key=_sort_key) + sorted(others, key=_sort_key)
    chosen: List[Dict[str, Any]] = []
    total_nf = 0
    for row in ordered:
        if len(chosen) >= max_clips:
            break
        chosen.append(row)
        total_nf += int(row.get("num_frames") or 0)
        if len(chosen) >= min_clips and total_nf >= target_total_frames:
            break
    return chosen


def _load_voice_keys(manifests: Sequence[Path]) -> set:
    keys: set = set()
    for tm in manifests:
        if not tm.is_file():
            continue
        with tm.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                vk = json.loads(line).get("voice_key")
                if vk:
                    keys.add(str(vk))
    return keys


def select_leak_free_refs(
    valid_manifest: Path,
    exclude_train_manifests: Sequence[Path],
    speakers: Sequence[str],
    min_clips: int = 2,
    max_clips: int = 3,
    target_total_frames: int = 375,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    primary_train = valid_manifest.parent / "train_manifest.jsonl"
    primary_excluded = _load_voice_keys([primary_train])
    all_excluded = primary_excluded | _load_voice_keys(exclude_train_manifests)

    joint_by_sp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    primary_by_sp: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    speaker_set = set(speakers)
    with valid_manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sp = str(row.get("speaker_id") or "")
            if sp not in speaker_set:
                continue
            lp = Path(str(row.get("latent_path") or ""))
            if not lp.is_file():
                continue
            vk = str(row.get("voice_key") or "")
            if not vk or vk not in all_excluded:
                joint_by_sp[sp].append(row)
            if not vk or vk not in primary_excluded:
                primary_by_sp[sp].append(row)

    selected: Dict[str, List[Dict[str, Any]]] = {}
    modes: Dict[str, str] = {}
    for sp in speakers:
        joint_chosen = _pick_clips_for_speaker(joint_by_sp.get(sp, []), max_clips, min_clips, target_total_frames)
        if len(joint_chosen) >= min_clips:
            selected[sp] = joint_chosen
            modes[sp] = "joint_leak_free"
            continue

        primary_chosen = _pick_clips_for_speaker(primary_by_sp.get(sp, []), max_clips, min_clips, target_total_frames)
        if len(primary_chosen) >= min_clips:
            print(
                f"[WARN] Speaker {sp!r} has only {len(joint_chosen)} joint leak-free ref(s); "
                f"falling back to primary valid split ({len(primary_chosen)} clip(s) unseen by primary & Base).",
                file=sys.stderr,
            )
            selected[sp] = primary_chosen
            modes[sp] = "primary_valid_fallback"
            continue

        raise RuntimeError(
            f"Speaker {sp!r} has only {len(primary_chosen)} held-out ref clip(s) even in {valid_manifest} "
            f"(joint={len(joint_chosen)})."
        )

    return selected, modes


def normalize_ja_for_cer(text: str) -> str:
    return _CER_STRIP_RE.sub("", text or "").strip()


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        cur = [i]
        for j, ch_b in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ch_a == ch_b else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def compute_cer(reference: str, hypothesis: str) -> Tuple[float, str, str]:
    ref_n = normalize_ja_for_cer(reference)
    hyp_n = normalize_ja_for_cer(hypothesis)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0, ref_n, hyp_n
    dist = levenshtein_distance(ref_n, hyp_n)
    return float(dist) / float(len(ref_n)), ref_n, hyp_n


def analyze_wav_acoustics(wav_path: Path) -> Dict[str, Optional[float]]:
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if nframes == 0 or sw not in (2, 4):
        return {"duration_sec": 0.0, "rms_db": None, "f0_median_hz": None}

    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0

    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)

    dur = float(len(x)) / float(max(1, sr))
    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    rms_db = float(20.0 * math.log10(max(rms, 1e-8)))

    # Autocorrelation F0 estimation (80 - 550 Hz) on 40ms frames / 10ms hop
    frame_len = max(1, int(0.040 * sr))
    hop_len = max(1, int(0.010 * sr))
    min_lag = max(1, int(sr / 550.0))
    max_lag = max(min_lag + 2, int(sr / 80.0))
    f0_vals: List[float] = []
    if len(x) >= frame_len and max_lag < frame_len:
        rms_thresh = max(0.01, rms * 0.25)
        for start in range(0, len(x) - frame_len + 1, hop_len):
            seg = x[start : start + frame_len]
            seg = seg - float(np.mean(seg))
            energy = float(np.dot(seg, seg))
            if math.sqrt(energy / float(frame_len)) < rms_thresh or energy < 1e-8:
                continue
            ac = np.correlate(seg, seg, mode="full")[frame_len - 1 :]
            if ac[0] <= 1e-8:
                continue
            ac_norm = ac / ac[0]
            sub = ac_norm[min_lag : max_lag + 1]
            if len(sub) == 0:
                continue
            peak_idx = int(np.argmax(sub))
            peak_val = float(sub[peak_idx])
            if peak_val >= 0.35:
                lag = min_lag + peak_idx
                # Parabolic refinement around peak_idx
                if 0 < peak_idx < len(sub) - 1:
                    y0, y1, y2 = float(sub[peak_idx - 1]), peak_val, float(sub[peak_idx + 1])
                    denom = 2.0 * (2.0 * y1 - y0 - y2)
                    if abs(denom) > 1e-8:
                        lag_f = float(lag) + (y2 - y0) / denom
                    else:
                        lag_f = float(lag)
                else:
                    lag_f = float(lag)
                if lag_f > 0:
                    f0_vals.append(float(sr) / lag_f)

    f0_median = float(np.median(f0_vals)) if f0_vals else None
    return {
        "duration_sec": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "f0_median_hz": round(f0_median, 1) if f0_median is not None else None,
    }


def _whisper_ld_env(whisper_py: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    prefix = whisper_py.expanduser().resolve().parent.parent
    lib_dirs: List[str] = []
    for pat in (
        "lib/python*/site-packages/torch/lib",
        "lib/python*/site-packages/nvidia/cudnn/lib",
        "lib/python*/site-packages/nvidia/cublas/lib",
    ):
        for cand in prefix.glob(pat):
            if cand.is_dir():
                lib_dirs.append(str(cand))
    if lib_dirs:
        existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + [d for d in existing if d not in lib_dirs])
    return env


def run_whisper_worker_main(in_json: Path, out_json: Path) -> int:
    """Invoked inside whisperx python environment with local HF cache only."""
    sys.path.insert(0, str(VCA_REPO))
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer
    from whisper_transcribe import transcribe_batch_gpu

    wav_strs: List[str] = json.loads(in_json.read_text(encoding="utf-8"))
    wav_paths: List[Optional[Path]] = [Path(s) if Path(s).is_file() else None for s in wav_strs]

    t0 = time.monotonic()
    model = WhisperModel("large-v3", device="cuda", compute_type="float16", local_files_only=True)
    tokenizer = Tokenizer(
        model.hf_tokenizer,
        model.model.is_multilingual,
        task="transcribe",
        language="ja",
    )
    ns = argparse.Namespace(
        batch_size=16,
        beam_size=3,
        language="ja",
        vad_filter=False,
        compression_ratio_threshold=2.2,
        no_speech_threshold=0.6,
        io_workers=8,
    )
    texts, fallbacks = transcribe_batch_gpu(model, tokenizer, wav_paths, ns)
    elapsed = time.monotonic() - t0
    print(f"[WHISPER] Transcribed {len(wav_paths)} clips in {elapsed:.1f}s (fallbacks={fallbacks})", flush=True)
    out_json.write_text(json.dumps(texts, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def run_batched_whisper(whisper_py: Path, wav_paths: List[Path], work_dir: Path) -> List[str]:
    if not whisper_py.is_file():
        raise FileNotFoundError(f"Whisper python not found: {whisper_py}")
    in_json = work_dir / "_whisper_inputs.json"
    out_json = work_dir / "_whisper_outputs.json"
    in_json.write_text(json.dumps([str(p) for p in wav_paths], ensure_ascii=False, indent=2), encoding="utf-8")
    if out_json.exists():
        out_json.unlink()

    env = _whisper_ld_env(whisper_py)
    cmd = [
        str(whisper_py),
        str(Path(__file__).resolve()),
        "--whisper-worker-in",
        str(in_json),
        "--whisper-worker-out",
        str(out_json),
    ]
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0 or not out_json.is_file():
        raise RuntimeError(f"Batched Whisper worker failed (rc={res.returncode})")
    return json.loads(out_json.read_text(encoding="utf-8"))


def decode_speaker_ref_wavs(
    codec: Any,
    selected_refs: Dict[str, List[Dict[str, Any]]],
    refs_dir: Path,
    save_wav_fn: Any,
    latent_dim: int = 32,
) -> Dict[str, str]:
    import torch

    refs_dir.mkdir(parents=True, exist_ok=True)
    ref_rel_paths: Dict[str, str] = {}
    for sp, rows in selected_refs.items():
        rel_str = f"refs/{sp}_ref.wav"
        ref_rel_paths[sp] = rel_str
        target_wav = refs_dir / f"{sp}_ref.wav"
        if target_wav.is_file():
            continue
        raw = torch.load(rows[0]["latent_path"], map_location="cpu", weights_only=True)
        if raw.ndim == 3 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim == 2 and raw.shape[0] == latent_dim and raw.shape[1] != latent_dim:
            raw = raw.transpose(0, 1).contiguous()
        audio = codec.decode_latent(raw.unsqueeze(0)).cpu()[0]
        save_wav_fn(target_wav, audio, int(codec.sample_rate))
    return ref_rel_paths


def synthesize_all_models(
    specs: Sequence[ModelSpec],
    prompts: Sequence[EvalPrompt],
    selected_refs: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    seed: int,
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], float]]:
    if str(IRODORI_REPO) not in sys.path:
        sys.path.insert(0, str(IRODORI_REPO))

    import torch
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest, save_wav

    refs_dir = out_dir / "refs"
    wavs_dir = out_dir / "wavs"
    ref_rel_paths: Dict[str, str] = {}
    synth_times: Dict[Tuple[str, str], float] = {}

    for m_idx, spec in enumerate(specs):
        print(f"\n[EVAL] ({m_idx + 1}/{len(specs)}) Loading model {spec.model_id}: {spec.checkpoint}", flush=True)
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(spec.checkpoint),
                model_device="cuda",
                codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                model_precision="bf16",
                codec_device="cuda",
                codec_precision="bf16",
                codec_deterministic_encode=True,
                codec_deterministic_decode=True,
                compile_model=False,
                compile_dynamic=False,
                split_rare_tokens=True,
            )
        )
        if not ref_rel_paths:
            ref_rel_paths = decode_speaker_ref_wavs(
                runtime.codec,
                selected_refs,
                refs_dir,
                save_wav,
                latent_dim=int(runtime.model_cfg.latent_dim),
            )

        m_dir = wavs_dir / spec.model_id
        m_dir.mkdir(parents=True, exist_ok=True)
        for p_idx, pr in enumerate(prompts, start=1):
            out_wav = m_dir / f"{pr.prompt_id}.wav"
            if out_wav.is_file():
                synth_times[(spec.model_id, pr.prompt_id)] = 0.0
                continue
            if spec.speaker_mode == "embed":
                assert spec.embed_dir is not None
                embed_file = spec.embed_dir / f"{pr.speaker_id}.speaker.safetensors"
                req = SamplingRequest(
                    text=pr.text,
                    caption=pr.caption,
                    ref_embed=str(embed_file),
                    cfg_scale_text=3.0,
                    cfg_scale_caption=3.0,
                    cfg_scale_speaker=5.0,
                    seed=seed,
                )
            elif spec.speaker_mode == "noref":
                req = SamplingRequest(
                    text=pr.text,
                    caption=pr.caption,
                    no_ref=True,
                    cfg_scale_text=3.0,
                    cfg_scale_caption=3.0,
                    cfg_scale_speaker=0.0,
                    seed=seed,
                )
            else:
                ref_latents = [str(r["latent_path"]) for r in selected_refs[pr.speaker_id]]
                req = SamplingRequest(
                    text=pr.text,
                    caption=pr.caption,
                    ref_latents=ref_latents,
                    cfg_scale_text=3.0,
                    cfg_scale_caption=3.0,
                    cfg_scale_speaker=5.0,
                    seed=seed,
                )
            t0 = time.monotonic()
            res = runtime.synthesize(req, log_fn=None)
            dt = time.monotonic() - t0
            save_wav(out_wav, res.audio, res.sample_rate)
            synth_times[(spec.model_id, pr.prompt_id)] = round(dt, 3)
            print(f"  [{spec.model_id} {p_idx}/{len(prompts)}] {pr.prompt_id}.wav ({dt:.2f}s)", flush=True)

        runtime.unload()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return ref_rel_paths, synth_times


def _mean_or_none(vals: Sequence[Optional[float]], digits: int = 2) -> Optional[float]:
    clean = [float(v) for v in vals if v is not None]
    if not clean:
        return None
    return round(sum(clean) / float(len(clean)), digits)


def build_summary_and_html(
    specs: Sequence[ModelSpec],
    prompts: Sequence[EvalPrompt],
    selected_refs: Dict[str, List[Dict[str, Any]]],
    ref_modes: Dict[str, str],
    ref_rel_paths: Dict[str, str],
    out_dir: Path,
    whisper_texts: Dict[Tuple[str, str], str],
    acoustics: Dict[Tuple[str, str], Dict[str, Optional[float]]],
    synth_times: Dict[Tuple[str, str], float],
    active_speakers: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    speakers_list = (
        list(active_speakers)
        if active_speakers is not None
        else [sp for sp in EVAL_SPEAKERS if sp in selected_refs]
    )
    records_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for spec in specs:
        for pr in prompts:
            key = (spec.model_id, pr.prompt_id)
            wt = whisper_texts.get(key, "")
            cer, ref_n, hyp_n = compute_cer(pr.text, wt)
            ac = acoustics.get(key, {})
            records_by_key[key] = {
                "model_id": spec.model_id,
                "prompt_id": pr.prompt_id,
                "section": pr.section,
                "speaker_id": pr.speaker_id,
                "category": pr.category,
                "category_label": pr.category_label,
                "wav_rel": f"wavs/{spec.model_id}/{pr.prompt_id}.wav",
                "text": pr.text,
                "caption": pr.caption,
                "whisper_text": wt,
                "norm_ref": ref_n,
                "norm_hyp": hyp_n,
                "cer": round(cer, 4),
                "duration_sec": ac.get("duration_sec"),
                "rms_db": ac.get("rms_db"),
                "f0_median_hz": ac.get("f0_median_hz"),
                "synth_sec": synth_times.get(key),
            }

    model_summaries: List[Dict[str, Any]] = []
    for spec in specs:
        m_recs = [records_by_key[(spec.model_id, pr.prompt_id)] for pr in prompts]
        sec_a = [r for r in m_recs if r["section"] == "A"]
        sec_b = [r for r in m_recs if r["section"] == "B"]
        per_sp_cer: Dict[str, Optional[float]] = {}
        for sp in speakers_list:
            sp_recs = [r for r in m_recs if r["speaker_id"] == sp]
            per_sp_cer[sp] = _mean_or_none([r["cer"] for r in sp_recs], digits=4)

        omote_recs = [r for r in sec_a if r["category"] == "omote"]
        ura_recs = [r for r in sec_a if r["category"] == "ura"]
        sw_recs = [r for r in sec_a if r["category"] == "switch"]
        omote_f0 = _mean_or_none([r["f0_median_hz"] for r in omote_recs], digits=1)
        ura_f0 = _mean_or_none([r["f0_median_hz"] for r in ura_recs], digits=1)
        switch_f0 = _mean_or_none([r["f0_median_hz"] for r in sw_recs], digits=1)
        f0_delta = (
            round(omote_f0 - ura_f0, 1)
            if (omote_f0 is not None and ura_f0 is not None)
            else None
        )
        fuyuko_metrics = {
            "omote_f0_mean_hz": omote_f0,
            "ura_f0_mean_hz": ura_f0,
            "switch_f0_mean_hz": switch_f0,
            "f0_delta_omote_minus_ura_hz": f0_delta,
            "omote_rms_mean_db": _mean_or_none([r["rms_db"] for r in omote_recs], digits=2),
            "ura_rms_mean_db": _mean_or_none([r["rms_db"] for r in ura_recs], digits=2),
            "switch_rms_mean_db": _mean_or_none([r["rms_db"] for r in sw_recs], digits=2),
            "omote_dur_mean_sec": _mean_or_none([r["duration_sec"] for r in omote_recs], digits=2),
            "ura_dur_mean_sec": _mean_or_none([r["duration_sec"] for r in ura_recs], digits=2),
        }
        model_summaries.append(
            {
                "model_id": spec.model_id,
                "display_name": spec.display_name,
                "speaker_mode": spec.speaker_mode,
                "checkpoint": str(spec.checkpoint),
                "overall_cer": _mean_or_none([r["cer"] for r in m_recs], digits=4),
                "section_a_cer": _mean_or_none([r["cer"] for r in sec_a], digits=4),
                "section_b_cer": _mean_or_none([r["cer"] for r in sec_b], digits=4),
                "per_speaker_cer": per_sp_cer,
                "fuyuko_persona_metrics": fuyuko_metrics,
            }
        )

    summary = {
        "eval_speakers": speakers_list,
        "num_prompts": len(prompts),
        "models": model_summaries,
        "references": {
            sp: {
                "selection_mode": ref_modes.get(sp, "joint_leak_free"),
                "ref_wav_rel": ref_rel_paths.get(sp, ""),
                "total_frames": sum(int(r.get("num_frames") or 0) for r in rows),
                "clips": [
                    {
                        "voice_key": r.get("voice_key"),
                        "num_frames": int(r.get("num_frames") or 0),
                        "latent_path": r.get("latent_path"),
                    }
                    for r in rows
                ],
            }
            for sp, rows in selected_refs.items()
        },
        "items": [records_by_key[(s.model_id, p.prompt_id)] for s in specs for p in prompts],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "index.html").write_text(
        render_html_report(specs, prompts, summary, records_by_key),
        encoding="utf-8",
    )
    return summary


def _fmt_pct(val: Optional[float]) -> str:
    return f"{val * 100.0:.2f}%" if val is not None else "—"


def _fmt_num(val: Optional[float], unit: str = "") -> str:
    return f"{val}{unit}" if val is not None else "—"


def render_html_report(
    specs: Sequence[ModelSpec],
    prompts: Sequence[EvalPrompt],
    summary: Dict[str, Any],
    records_by_key: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    e = html.escape
    model_rows_html = []
    for ms in summary["models"]:
        fm = ms["fuyuko_persona_metrics"]
        model_rows_html.append(
            f"<tr><td><strong>{e(ms['display_name'])}</strong><div class='sub'>{e(ms['model_id'])}</div></td>"
            f"<td>{_fmt_pct(ms['overall_cer'])}</td>"
            f"<td>{_fmt_pct(ms['section_a_cer'])}</td>"
            f"<td>{_fmt_pct(ms['section_b_cer'])}</td>"
            f"<td>{_fmt_num(fm['omote_f0_mean_hz'], ' Hz')}</td>"
            f"<td>{_fmt_num(fm['ura_f0_mean_hz'], ' Hz')}</td>"
            f"<td><strong>{_fmt_num(fm['f0_delta_omote_minus_ura_hz'], ' Hz')}</strong></td>"
            f"<td>{_fmt_num(fm['omote_rms_mean_db'], ' dB')} / {_fmt_num(fm['ura_rms_mean_db'], ' dB')}</td></tr>"
        )

    sec_a_prompts = [p for p in prompts if p.section == "A"]
    sec_a_rows = []
    for p in sec_a_prompts:
        cells = []
        for s in specs:
            r = records_by_key[(s.model_id, p.prompt_id)]
            badge_cls = "good" if r["cer"] <= 0.05 else ("warn" if r["cer"] <= 0.15 else "bad")
            cells.append(
                f"<td>"
                f"<audio controls preload='none' src='{e(r['wav_rel'])}'></audio>"
                f"<div class='metrics'>"
                f"<span class='badge {badge_cls}'>CER {_fmt_pct(r['cer'])}</span>"
                f"<span class='badge'>F0 {_fmt_num(r['f0_median_hz'], 'Hz')}</span>"
                f"<span class='badge'>RMS {_fmt_num(r['rms_db'], 'dB')}</span>"
                f"<span class='badge'>{_fmt_num(r['duration_sec'], 's')}</span>"
                f"</div>"
                f"<div class='asr'>ASR: {e(r['whisper_text'] or '(empty)')}</div>"
                f"</td>"
            )
        sec_a_rows.append(
            f"<tr>"
            f"<td class='meta'>"
            f"<div class='tag'>{e(p.category_label)}</div>"
            f"<div class='txt'>{e(p.text)}</div>"
            f"<div class='cap'>{e(p.caption)}</div>"
            f"</td>"
            + "".join(cells)
            + "</tr>"
        )

    sec_b_rows = []
    sec_b_by_sp: Dict[str, List[EvalPrompt]] = defaultdict(list)
    for p in prompts:
        if p.section == "B":
            sec_b_by_sp[p.speaker_id].append(p)

    for sp in summary["eval_speakers"]:
        if not sec_b_by_sp.get(sp):
            continue
        ref_info = summary["references"][sp]
        sp_header = (
            f"<tr class='sp-sep'><td colspan='{len(specs) + 1}'>"
            f"<div class='sp-bar'>"
            f"<span><strong>{e(SPEAKER_NAMES_JA.get(sp, sp))}</strong> (<code>{e(sp)}</code>, held-out ref={ref_info['total_frames']/25.0:.1f}s)</span>"
            f"<audio controls preload='none' src='{e(ref_info['ref_wav_rel'])}'></audio>"
            f"</div></td></tr>"
        )
        sec_b_rows.append(sp_header)
        for p in sec_b_by_sp[sp]:
            cells = []
            for s in specs:
                r = records_by_key[(s.model_id, p.prompt_id)]
                badge_cls = "good" if r["cer"] <= 0.05 else ("warn" if r["cer"] <= 0.15 else "bad")
                cells.append(
                    f"<td>"
                    f"<audio controls preload='none' src='{e(r['wav_rel'])}'></audio>"
                    f"<div class='metrics'>"
                    f"<span class='badge {badge_cls}'>CER {_fmt_pct(r['cer'])}</span>"
                    f"<span class='badge'>F0 {_fmt_num(r['f0_median_hz'], 'Hz')}</span>"
                    f"<span class='badge'>RMS {_fmt_num(r['rms_db'], 'dB')}</span>"
                    f"<span class='badge'>{_fmt_num(r['duration_sec'], 's')}</span>"
                    f"</div>"
                    f"<div class='asr'>ASR: {e(r['whisper_text'] or '(empty)')}</div>"
                    f"</td>"
                )
            sec_b_rows.append(
                f"<tr>"
                f"<td class='meta'>"
                f"<div class='tag'>{e(p.category_label)} ({e(p.category)})</div>"
                f"<div class='txt'>{e(p.text)}</div>"
                f"<div class='cap'>{e(p.caption)}</div>"
                f"</td>"
                + "".join(cells)
                + "</tr>"
            )

    model_headers = "".join(f"<th>{e(s.display_name)}</th>" for s in specs)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SC29 3-Way Evaluation Report</title>
<style>
  :root {{ color-scheme: dark; --bg:#0d1117; --card:#161b22; --line:#30363d; --text:#e6edf3; --muted:#8b949e; --accent:#58a6ff; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--text); font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; }}
  h1,h2 {{ margin:0 0 12px; }}
  h2 {{ margin-top:28px; border-bottom:1px solid var(--line); padding-bottom:6px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin-bottom:20px; overflow-x:auto; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th,td {{ border:1px solid var(--line); padding:8px 10px; vertical-align:top; text-align:left; }}
  th {{ background:#1f242c; position:sticky; top:0; }}
  audio {{ width:230px; height:32px; display:block; margin-bottom:6px; }}
  .meta {{ min-width:250px; max-width:340px; }}
  .tag {{ font-size:11px; color:var(--accent); font-weight:600; margin-bottom:4px; }}
  .txt {{ font-size:13px; margin-bottom:4px; }}
  .cap {{ font-size:12px; color:var(--muted); }}
  .metrics {{ display:flex; flex-wrap:wrap; gap:4px; margin-bottom:4px; }}
  .badge {{ font-size:11px; padding:2px 6px; border-radius:4px; background:#21262d; color:#c9d1d9; }}
  .badge.good {{ background:#163822; color:#7ee787; }}
  .badge.warn {{ background:#3a2f12; color:#f2cc60; }}
  .badge.bad {{ background:#441c1f; color:#ff7b72; }}
  .asr {{ font-size:11px; color:var(--muted); word-break:break-word; }}
  .sub {{ font-size:11px; color:var(--muted); }}
  .sp-sep td {{ background:#1b2129; }}
  .sp-bar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }}
</style>
</head>
<body>
<h1>Shiny Colors 29-Character Irodori-TTS 3-Way Evaluation</h1>
<p class="sub">Models: {", ".join(e(s.display_name) for s in specs)} | Prompts: {len(prompts)} (9 Fuyuko Dual-Persona + 24 Multi-Character) | Zero Train-Set Leakage Held-Out Refs</p>

<h2>Summary Metrics</h2>
<div class="card">
<table>
  <thead><tr><th>Model</th><th>Overall CER</th><th>Sec A Fuyuko CER</th><th>Sec B 8-Char CER</th><th>Omote Mean F0</th><th>Ura Mean F0</th><th>F0 Δ (Omote − Ura)</th><th>RMS (Omote / Ura)</th></tr></thead>
  <tbody>{"".join(model_rows_html)}</tbody>
</table>
</div>

<h2>Section A — 冬優子双人格切换矩阵 (3 句 × 3 触发词)</h2>
<div class="card">
<table>
  <thead><tr><th>Prompt &amp; Caption</th>{model_headers}</tr></thead>
  <tbody>{"".join(sec_a_rows)}</tbody>
</table>
</div>

<h2>Section B — 8 角色多情绪对比评测</h2>
<div class="card">
<table>
  <thead><tr><th>Character &amp; Emotion</th>{model_headers}</tr></thead>
  <tbody>{"".join(sec_b_rows)}</tbody>
</table>
</div>
</body>
</html>
"""


def serve_directory(out_dir: Path, port: int) -> None:
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(out_dir), **kwargs)

    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"[EVAL] Serving {out_dir} at http://127.0.0.1:{port}/index.html", flush=True)
    srv.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified SC29 3-way evaluation runner")
    parser.add_argument("--checkpoint", default=None, help="Primary checkpoint path (or label=path)")
    parser.add_argument("--extra-checkpoint", action="append", default=[], help="Additional checkpoint path (or label=path)")
    parser.add_argument("--base-checkpoint", default=str(DEFAULT_BASE_CKPT), help="Base checkpoint path (or 'none' to skip)")
    parser.add_argument("--valid-manifest", type=Path, default=None, help="Validation manifest JSONL")
    parser.add_argument("--exclude-train-manifest", action="append", default=[], help="Train manifest JSONL(s) to exclude for leak-free refs")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--whisper-python", type=Path, default=DEFAULT_WHISPER_PY)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speakers", default=None, help="Comma-separated subset of speakers to evaluate (e.g. fuyuko,mano,luca)")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--serve-only", action="store_true", help="Serve existing out-dir HTML report without running synthesis/ASR")
    parser.add_argument("--dry-run", action="store_true", help="Validate refs, prompts, and checkpoints without GPU")
    parser.add_argument("--whisper-worker-in", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--whisper-worker-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.whisper_worker_in is not None and args.whisper_worker_out is not None:
        return run_whisper_worker_main(args.whisper_worker_in, args.whisper_worker_out)

    if args.serve_only:
        if args.port is None:
            raise ValueError("--serve-only requires --port")
        index_html = args.out_dir / "index.html"
        if not index_html.is_file():
            raise FileNotFoundError(f"Missing evaluation report to serve: {index_html}")
        serve_directory(args.out_dir, int(args.port))
        return 0

    specs = resolve_model_specs(args)
    if args.speakers:
        active_speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
        unknown = [s for s in active_speakers if s not in EVAL_SPEAKERS]
        if unknown:
            raise ValueError(f"Unknown --speakers: {unknown}. Allowed: {EVAL_SPEAKERS}")
    else:
        active_speakers = list(EVAL_SPEAKERS)

    active_set = set(active_speakers)
    prompts = [p for p in build_prompt_suite() if p.speaker_id in active_set]
    for s in specs:
        if s.speaker_mode == "embed":
            assert s.embed_dir is not None
            missing = [
                str(s.embed_dir / f"{sp}.speaker.safetensors")
                for sp in active_speakers
                if not (s.embed_dir / f"{sp}.speaker.safetensors").is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing speaker inversion embedding(s) for {s.model_id}: {missing}"
                )

    valid_manifest = resolve_valid_manifest(args.valid_manifest)
    exclude_paths = list(DEFAULT_TRAIN_EXCLUDES) + [Path(p).expanduser() for p in args.exclude_train_manifest]
    selected_refs, ref_modes = select_leak_free_refs(valid_manifest, exclude_paths, active_speakers)

    print(f"[DRY-RUN CHECK] models={len(specs)}: {[f'{s.model_id}({s.speaker_mode})={s.checkpoint.name}' for s in specs]}")
    print(f"[DRY-RUN CHECK] valid_manifest={valid_manifest} | speakers={len(active_speakers)} | prompts={len(prompts)} (A={sum(1 for p in prompts if p.section=='A')}, B={sum(1 for p in prompts if p.section=='B')})")
    for sp in active_speakers:
        clips = selected_refs[sp]
        frames = sum(int(r.get("num_frames") or 0) for r in clips)
        print(f"  ref[{sp}] ({ref_modes[sp]}): {len(clips)} held-out clips ({frames} frames / {frames/25.0:.1f}s)")

    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ref_rel_paths, synth_times = synthesize_all_models(specs, prompts, selected_refs, args.out_dir, args.seed)

    wav_order: List[Tuple[str, str, Path]] = []
    for s in specs:
        for p in prompts:
            wav_order.append((s.model_id, p.prompt_id, args.out_dir / "wavs" / s.model_id / f"{p.prompt_id}.wav"))

    print(f"\n[EVAL] Running local batched Whisper large-v3 over {len(wav_order)} WAV(s)...", flush=True)
    raw_texts = run_batched_whisper(args.whisper_python, [item[2] for item in wav_order], args.out_dir)
    whisper_texts = {(m_id, p_id): txt for (m_id, p_id, _), txt in zip(wav_order, raw_texts)}

    acoustics: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
    for m_id, p_id, wp in wav_order:
        acoustics[(m_id, p_id)] = analyze_wav_acoustics(wp)

    summary = build_summary_and_html(
        specs,
        prompts,
        selected_refs,
        ref_modes,
        ref_rel_paths,
        args.out_dir,
        whisper_texts,
        acoustics,
        synth_times,
        active_speakers=active_speakers,
    )
    print("\n[EVAL DONE] Summary:")
    for ms in summary["models"]:
        fm = ms["fuyuko_persona_metrics"]
        print(
            f"  {ms['model_id']}: overall_cer={ms['overall_cer']*100:.2f}% | "
            f"secA_cer={ms['section_a_cer']*100:.2f}% | secB_cer={ms['section_b_cer']*100:.2f}% | "
            f"fuyuko_F0(omote/ura/delta)={fm['omote_f0_mean_hz']}/{fm['ura_f0_mean_hz']}/{fm['f0_delta_omote_minus_ura_hz']} Hz"
        )
    print(f"[EVAL] Report written to: {args.out_dir / 'index.html'} and {args.out_dir / 'summary.json'}")

    if args.port is not None:
        serve_directory(args.out_dir, int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
