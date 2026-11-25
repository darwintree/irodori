#!/usr/bin/env python3
"""SC29 MeanFlow CPU Inference & Acceptance Workbench (HTTP/1.1).

Serves a local two-tab web workbench on port 8788 (default):
  1. Interactive CPU Inference Studio:
     - All 29 Shiny Colors characters via 49 KB --ref-embed (s8000)
     - clarity_fix_1500 MeanFlow (4/6/8/12/16 steps, CPU default or GPU)
     - Optional Hybrid mode (clarity_fix MF 3-step 1.0->0.25 + Teacher s8000 RF 5-step 0.25->0.0)
     - Optional Teacher s8000 RF-40 baseline (ceiling reference)
     - Optional 20k_anneal MeanFlow if extracted under outputs/sc29_v1_meanflow_20k/
     - Real-time latency, RTF, and stage timing breakdown + persistent history
  2. Acceptance Comparison Matrix:
     - 10 curated diagnostic cases (Hinana long drag, Fuyuko personas, Chiyuki/Hazuki long lines,
       Madoka, Luca) with side-by-side playback across clarity_fix 4-step CPU, 8-step CPU,
       Hybrid MF3+RF5 CPU, and Teacher s8000 RF-40.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("IRODORI_SDPA_CUDNN", "0")

import argparse
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import secrets
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
import uuid

ROOT = Path(__file__).resolve().parent.parent
IRODORI_REPO = ROOT / "Irodori-TTS"
if str(IRODORI_REPO) not in sys.path:
    sys.path.insert(0, str(IRODORI_REPO))

import torch  # noqa: E402

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    save_wav,
)
from irodori_tts.meanflow import EncodedConditions  # noqa: E402
from irodori_tts.rf import draw_initial_noise  # noqa: E402

SPEAKERS_DIR = ROOT / "outputs/sc29_v1_speakers_s8000"
CLARITY_CKPT = ROOT / "outputs/sc29_v1_meanflow_clarity_fix/model_meanflow_s8000_clarity_fix.safetensors"
MF20K_CKPT = ROOT / "outputs/sc29_v1_meanflow_20k/model_meanflow_s8000_20k.safetensors"
TEACHER_CKPT = ROOT / "outputs/sc29_v1_full/model_s8000.safetensors"
WORKBENCH_DIR = ROOT / "outputs/sc29_v1_cpu_workbench"
HISTORY_DIR = WORKBENCH_DIR / "history"
HISTORY_INDEX = HISTORY_DIR / "index.json"
BENCH_DIR = WORKBENCH_DIR / "bench"
CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"

# All 29 Shiny Colors speakers grouped by unit
SPEAKER_CATALOG: list[dict[str, str]] = [
    # Illumination Stars
    {"id": "mano", "ja": "櫻木真乃", "zh": "樱木真乃", "unit": "illumination STARS"},
    {"id": "hiori", "ja": "風野灯織", "zh": "风野灯织", "unit": "illumination STARS"},
    {"id": "meguru", "ja": "八宮めぐる", "zh": "八宫巡", "unit": "illumination STARS"},
    # L'Antica
    {"id": "kogane", "ja": "月岡恋鐘", "zh": "月冈恋钟", "unit": "L'Antica"},
    {"id": "mamimi", "ja": "田中摩美々", "zh": "田中摩美美", "unit": "L'Antica"},
    {"id": "sakuya", "ja": "白瀬咲耶", "zh": "白濑咲耶", "unit": "L'Antica"},
    {"id": "yuika", "ja": "三峰結華", "zh": "三峰结华", "unit": "L'Antica"},
    {"id": "kiriko", "ja": "幽谷霧子", "zh": "幽谷雾子", "unit": "L'Antica"},
    # Houkago Climax Girls
    {"id": "kaho", "ja": "小宮果穂", "zh": "小宫果穗", "unit": "放課後クライマックスガールズ"},
    {"id": "chiyoko", "ja": "園田智代子", "zh": "园田智代子", "unit": "放課後クライマックスガールズ"},
    {"id": "juri", "ja": "西城樹里", "zh": "西城树里", "unit": "放課後クライマックスガールズ"},
    {"id": "rinze", "ja": "杜野凛世", "zh": "杜野凛世", "unit": "放課後クライマックスガールズ"},
    {"id": "natsuha", "ja": "有栖川夏葉", "zh": "有栖川夏叶", "unit": "放課後クライマックスガールズ"},
    # Alstroemeria
    {"id": "amana", "ja": "大崎甘奈", "zh": "大崎甘奈", "unit": "ALSTROEMERIA"},
    {"id": "tenka", "ja": "大崎甜花", "zh": "大崎甜花", "unit": "ALSTROEMERIA"},
    {"id": "chiyuki", "ja": "桑山千雪", "zh": "桑山千雪", "unit": "ALSTROEMERIA"},
    # Straylight
    {"id": "asahi", "ja": "芹沢あさひ", "zh": "芹泽朝日", "unit": "Straylight"},
    {"id": "fuyuko", "ja": "黛冬優子", "zh": "黛冬优子", "unit": "Straylight"},
    {"id": "mei", "ja": "和泉愛依", "zh": "和泉爱依", "unit": "Straylight"},
    # noctchill
    {"id": "toru", "ja": "浅倉透", "zh": "浅仓透", "unit": "noctchill"},
    {"id": "madoka", "ja": "樋口円香", "zh": "樋口圆香", "unit": "noctchill"},
    {"id": "koito", "ja": "福丸小糸", "zh": "福丸小糸", "unit": "noctchill"},
    {"id": "hinana", "ja": "市川雛菜", "zh": "市川雏菜", "unit": "noctchill"},
    # SHHis
    {"id": "nichika", "ja": "七草にちか", "zh": "七草日花", "unit": "SHHis"},
    {"id": "mikoto", "ja": "緋田美琴", "zh": "绯田美琴", "unit": "SHHis"},
    # CoMETIK
    {"id": "luca", "ja": "斑鳩ルカ", "zh": "斑鸠路加", "unit": "CoMETIK"},
    {"id": "hana", "ja": "鈴木羽那", "zh": "铃木羽那", "unit": "CoMETIK"},
    {"id": "haruki", "ja": "郁田はるき", "zh": "郁田春希", "unit": "CoMETIK"},
    # 283 Pro Staff
    {"id": "hazuki", "ja": "七草はづき", "zh": "七草叶月", "unit": "283 Pro Staff"},
]

SPEAKER_BY_ID: dict[str, dict[str, str]] = {item["id"]: item for item in SPEAKER_CATALOG}

BENCH_CASES: list[dict[str, Any]] = [
    {
        "id": "hinana_drag",
        "title": "市川雛菜 · 高曲率拖长音",
        "tag": "高曲率拖音",
        "speaker_id": "hinana",
        "text": "やは〜、プロデューサー今日もかっこいい〜。雛菜、すっごくしあわせ〜！",
        "caption": "甘く弾んだ明るい高い声で、語尾を可愛らしく伸ばしながらとても嬉しそうに話している。",
        "seed": 42,
    },
    {
        "id": "fuyuko_omote",
        "title": "黛冬優子 · 表（営業スマイル）",
        "tag": "双人格 · 表",
        "speaker_id": "fuyuko",
        "text": "プロデューサーさん、今日のお仕事もふゆにお任せくださいねっ！",
        "caption": "猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。",
        "seed": 42,
    },
    {
        "id": "fuyuko_ura",
        "title": "黛冬優子 · 裏（素の本音）",
        "tag": "双人格 · 裏",
        "speaker_id": "fuyuko",
        "text": "もう……あんた、ちゃんとふゆの話聞いてたんですか？",
        "caption": "素の本音口調で、低めの少しぶっきらぼうなトーンで呆れたように話している。",
        "seed": 42,
    },
    {
        "id": "fuyuko_angry",
        "title": "黛冬優子 · 本音激昂",
        "tag": "强情绪 · 爆发",
        "speaker_id": "fuyuko",
        "text": "ちょっとあんた、本気でそんなこと言ってんの！？信じらんないっ！",
        "caption": "素の本音口調で、苛立ちを抑えきれず強く鋭い声で言い放っている。",
        "seed": 42,
    },
    {
        "id": "fuyuko_whisper",
        "title": "黛冬優子 · 气声耳语",
        "tag": "气声 · 清晰度",
        "speaker_id": "fuyuko",
        "text": "……しっ、静かにして。誰も見てないから……今だけ、素のままでいさせてよ。",
        "caption": "息を多めに含ませた控えめな囁き声で、少し照れながら静かに語りかけている。",
        "seed": 42,
    },
    {
        "id": "chiyuki_whisper_long",
        "title": "桑山千雪 · 温柔耳语长语音",
        "tag": "长语音 · 清晰度",
        "speaker_id": "chiyuki",
        "text": "プロデューサーさん、今日も本当にお疲れ様でした……ねえ、そんなに頑張りすぎないでくださいね。たまには、私のそばでゆっくり息を抜いて……ほら、温かい紅茶、淹れておきましたから。",
        "caption": "お姉さんのような優しく包み込む柔らかな囁き声で、息を少し含ませながらゆったりと労わっている。",
        "seed": 42,
    },
    {
        "id": "chiyuki_serious_long",
        "title": "桑山千雪 · 严肃劝阻长语音",
        "tag": "长语音 · 严肃",
        "speaker_id": "chiyuki",
        "text": "待ってください、プロデューサーさん。このまま無理を続けたら、本当に倒れてしまいます。みんなの笑顔を守るためにも、今だけは私の言うことを聞いて、ちゃんと休んでください。",
        "caption": "普段の優しさの奥に強い決意を込めた凛とした真剣な口調で、はっきりと言い聞かせるように話している。",
        "seed": 42,
    },
    {
        "id": "hazuki_night_long",
        "title": "七草はづき · 深夜叹气关怀长语音",
        "tag": "长语音 · 叹气",
        "speaker_id": "hazuki",
        "text": "はぁ……もうこんな時間なんですね。プロデューサーさん、まだ起きてらっしゃったんですか？お仕事熱心なのは素敵ですけど、今夜はこのくらいにして、明日に備えましょうね。",
        "caption": "少し疲れを滲ませた優しいため息から始まり、おっとりとした穏やかな声で気遣うように語りかけている。",
        "seed": 42,
    },
    {
        "id": "madoka_cold",
        "title": "樋口円香 · 冷淡皮肉",
        "tag": "低音 · 咬字",
        "speaker_id": "madoka",
        "text": "ふっ……相変わらずおめでたいですね、ミスター・好青年。",
        "caption": "少し皮肉混じりに口角を上げながら、静かで落ち着いた冷ややかな低い声で話している。",
        "seed": 42,
    },
    {
        "id": "luca_low_gentle",
        "title": "斑鳩ルカ · 低音防串角验证",
        "tag": "低音 · Embed锁声",
        "speaker_id": "luca",
        "text": "……あんたさ、今日だけは、ちょっとありがとよ。",
        "caption": "肩の力を抜いた低く少し柔らかな声で、控えめながら素直に感謝を伝えている。",
        "seed": 42,
    },
]

BENCH_CASE_BY_ID = {c["id"]: c for c in BENCH_CASES}

BENCH_COLUMNS: list[dict[str, Any]] = [
    {
        "id": "clarity_mf4_cpu",
        "label": "clarity_fix 4步 (CPU)",
        "short": "CF 4步 CPU",
        "model": "clarity_fix",
        "device": "cpu",
        "num_steps": 4,
        "badge": "cpu",
    },
    {
        "id": "clarity_mf8_cpu",
        "label": "clarity_fix 8步 (CPU)",
        "short": "CF 8步 CPU",
        "model": "clarity_fix",
        "device": "cpu",
        "num_steps": 8,
        "badge": "cpu8",
    },
    {
        "id": "hybrid_3_5_cpu",
        "label": "混合 MF3+RF5 (CPU 8 NFE)",
        "short": "混合 3+5 CPU",
        "model": "hybrid_clarity_teacher",
        "device": "cpu",
        "num_steps": 8,
        "auto_chunk": False,
        "badge": "hybrid",
    },
    {
        "id": "clarity_mf4_chunk_cpu",
        "label": "clarity_fix 4步 分句拼接 (CPU)",
        "short": "CF 4步 分句",
        "model": "clarity_fix",
        "device": "cpu",
        "num_steps": 4,
        "auto_chunk": True,
        "max_chars": 45,
        "pause_ms": 220,
        "badge": "chunk4",
    },
    {
        "id": "clarity_mf8_chunk_cpu",
        "label": "clarity_fix 8步 分句拼接 (CPU)",
        "short": "CF 8步 分句",
        "model": "clarity_fix",
        "device": "cpu",
        "num_steps": 8,
        "auto_chunk": True,
        "max_chars": 45,
        "pause_ms": 220,
        "badge": "chunk8",
    },
    {
        "id": "teacher_rf40",
        "label": "Teacher s8000 RF-40 (天花板)",
        "short": "Teacher RF-40",
        "model": "teacher_s8000",
        "device": "auto_gpu",
        "num_steps": 40,
        "auto_chunk": False,
        "badge": "teacher",
    },
]
BENCH_COL_BY_ID = {c["id"]: c for c in BENCH_COLUMNS}

_STRONG_SPLIT_RE = re.compile(r"([。！？!?]+[」』）】\"']*)")
_SOFT_PAUSE_RE = re.compile(r"((?:……+|…+|――+|—+|\n+)[」』）】\"']*)")
_COMMA_SPLIT_RE = re.compile(r"([、，,])")


def _split_keep_delim(text: str, pattern: re.Pattern[str]) -> list[str]:
    parts = pattern.split(text)
    out: list[str] = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if pattern.fullmatch(p):
            buf += p
            if buf.strip():
                out.append(buf.strip())
            buf = ""
        else:
            if buf.strip():
                out.append(buf.strip())
            buf = p
    if buf.strip():
        out.append(buf.strip())
    return out


def split_japanese_chunks(text: str, *, max_chars: int = 45, min_chars: int = 12) -> list[str]:
    """Split long Japanese utterance along natural punctuation into 4-9s sweet-spot chunks."""
    clean = re.sub(r"\s+", " ", str(text or "").strip())
    if not clean:
        return []
    max_c = max(20, int(max_chars))
    min_c = max(6, min(int(min_chars), max_c // 2))

    # 1) Split on strong sentence endings (。！？!?)
    initial = _split_keep_delim(clean, _STRONG_SPLIT_RE)

    # 2) Break any overly long segment on ellipsis/dashes first, then commas
    refined: list[str] = []
    for seg in initial:
        if len(seg) <= max_c:
            refined.append(seg)
            continue
        sub_segs = _split_keep_delim(seg, _SOFT_PAUSE_RE)
        # Re-group soft pieces greedily up to max_c
        grouped: list[str] = []
        for s in sub_segs:
            if grouped and len(grouped[-1]) + len(s) <= max_c:
                grouped[-1] += s
            else:
                grouped.append(s)
        # If any piece is still > max_c, split near midpoint on comma
        for g in grouped:
            if len(g) <= max_c:
                refined.append(g)
                continue
            comma_pieces = _split_keep_delim(g, _COMMA_SPLIT_RE)
            if len(comma_pieces) <= 1:
                refined.append(g)
                continue
            cur = ""
            for cp in comma_pieces:
                if cur and len(cur) + len(cp) > max_c and len(cur) >= min_c:
                    refined.append(cur)
                    cur = cp
                else:
                    cur += cp
            if cur:
                refined.append(cur)

    # 3) Merge ultra-short fragments (e.g. "はぁ……" or "……！") with neighbor
    merged: list[str] = []
    i = 0
    while i < len(refined):
        seg = refined[i]
        if len(seg) < min_c:
            if i + 1 < len(refined) and len(seg) + len(refined[i + 1]) <= max_c + 10:
                refined[i + 1] = seg + refined[i + 1]
                i += 1
                continue
            if merged and len(merged[-1]) + len(seg) <= max_c + 10:
                merged[-1] += seg
                i += 1
                continue
        merged.append(seg)
        i += 1

    return merged or [clean]


def _trim_chunk_edges(
    wav: torch.Tensor,
    sample_rate: int,
    *,
    trim_lead: bool,
    trim_tail: bool,
    threshold: float = 0.002,
    lead_margin_ms: int = 50,
    tail_margin_ms: int = 80,
) -> torch.Tensor:
    """Conservatively trim dead silence padding at internal chunk boundaries."""
    w = wav.detach().to(device="cpu", dtype=torch.float32)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    n = int(w.shape[-1])
    if n == 0:
        return w
    amp = w.abs().max(dim=0).values
    nz = torch.nonzero(amp > threshold, as_tuple=False).flatten()
    if nz.numel() == 0:
        return w
    start = 0
    end = n
    if trim_lead:
        lead_pad = int(sample_rate * lead_margin_ms / 1000.0)
        start = max(0, int(nz[0].item()) - lead_pad)
    if trim_tail:
        tail_pad = int(sample_rate * tail_margin_ms / 1000.0)
        end = min(n, int(nz[-1].item()) + 1 + tail_pad)
    if end <= start:
        return w
    return w[:, start:end].contiguous()


def stitch_audio_chunks(
    audios: list[torch.Tensor],
    sample_rate: int,
    *,
    pause_ms: int = 220,
    chunk_texts: list[str] | None = None,
) -> torch.Tensor:
    """Stitch synthesized chunks with gentle edge fades and natural punctuation pauses."""
    if not audios:
        return torch.zeros((1, 1), dtype=torch.float32)
    if len(audios) == 1:
        w = audios[0].detach().to(device="cpu", dtype=torch.float32)
        return w.unsqueeze(0) if w.ndim == 1 else w

    fade_samples = max(1, int(sample_rate * 0.012))  # 12ms anti-click fade
    pieces: list[torch.Tensor] = []
    total_chunks = len(audios)
    for idx, raw in enumerate(audios):
        w = _trim_chunk_edges(
            raw,
            sample_rate,
            trim_lead=(idx > 0),
            trim_tail=(idx < total_chunks - 1),
        ).clone()
        n = int(w.shape[-1])
        f_len = min(fade_samples, n // 4)
        if f_len > 1:
            if idx > 0:
                ramp = torch.linspace(0.0, 1.0, f_len, dtype=w.dtype)
                w[:, :f_len] *= ramp
            if idx < total_chunks - 1:
                ramp = torch.linspace(1.0, 0.0, f_len, dtype=w.dtype)
                w[:, -f_len:] *= ramp
        pieces.append(w)

        if idx < total_chunks - 1:
            gap_ms = max(0, int(pause_ms))
            if chunk_texts and idx < len(chunk_texts):
                tail_txt = chunk_texts[idx].rstrip("」』）】\"' ")
                if tail_txt.endswith(("…", "—", "―", "、", "，", ",")):
                    gap_ms = int(gap_ms * 0.72)
            gap_samples = int(sample_rate * gap_ms / 1000.0)
            if gap_samples > 0:
                pieces.append(torch.zeros((w.shape[0], gap_samples), dtype=w.dtype))

    return torch.cat(pieces, dim=-1)

CAPTION_CHIPS: list[tuple[str, str]] = [
    ("明亮元气", "明るく弾んだ口調で、心から嬉しそうに元気よく話している。"),
    ("温柔安抚", "穏やかで優しい口調で、相手を気遣いながら静かに語りかけている。"),
    ("轻声耳语", "息を多めに含ませた控えめな囁き声で、そっと静かに話している。"),
    ("冷淡低音", "感情を抑えた冷たく淡々とした低い口調で、落ち着いて話している。"),
    ("含泪颤抖", "涙を堪えながら声を小さく震わせ、今にも泣き出しそうな弱い声で訴えている。"),
    ("怒气爆发", "苛立ちと怒りを抑えきれず、強く鋭い声で勢いよく言い放っている。"),
    ("冬优子·表", "猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。"),
    ("冬优子·裏", "素の本音口調で、低めの少しぶっきらぼうなトーンで呆れたように話している。"),
]


class EngineManager:
    """Thread-safe model cache and CPU/GPU/Hybrid synthesizer."""

    def __init__(self, default_threads: int = 10) -> None:
        self._lock = threading.Lock()
        self._runtimes: dict[tuple[str, str], InferenceRuntime] = {}
        self._default_threads = int(default_threads)
        self._preload_state = "idle"
        self._preload_error: str | None = None
        self._preload_sec: float | None = None
        self._cuda_available = bool(torch.cuda.is_available())
        self._cuda_name = torch.cuda.get_device_name(0) if self._cuda_available else None
        self._bench_job_lock = threading.Lock()
        self._bench_job: dict[str, Any] = {
            "running": False,
            "done": 0,
            "total": 0,
            "current": "",
            "error": None,
        }
        torch.set_num_threads(self._default_threads)

    def available_models(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        if CLARITY_CKPT.is_file():
            models.append({
                "id": "clarity_fix",
                "name": "clarity_fix_1500 (MeanFlow 清晰度补训)",
                "flow": "meanflow",
                "default_steps": 4,
                "path": str(CLARITY_CKPT),
            })
        if MF20K_CKPT.is_file():
            models.append({
                "id": "mf_20k",
                "name": "20k_anneal (MeanFlow 20k 退火)",
                "flow": "meanflow",
                "default_steps": 4,
                "path": str(MF20K_CKPT),
            })
        if CLARITY_CKPT.is_file() and TEACHER_CKPT.is_file():
            models.append({
                "id": "hybrid_clarity_teacher",
                "name": "混合推理: clarity_fix MF 3步 + Teacher RF 5步 (8 NFE)",
                "flow": "hybrid",
                "default_steps": 8,
                "path": f"{CLARITY_CKPT.name} + {TEACHER_CKPT.name}",
            })
        if MF20K_CKPT.is_file() and TEACHER_CKPT.is_file():
            models.append({
                "id": "hybrid_20k_teacher",
                "name": "混合推理: 20k MF 3步 + Teacher RF 5步 (8 NFE)",
                "flow": "hybrid",
                "default_steps": 8,
                "path": f"{MF20K_CKPT.name} + {TEACHER_CKPT.name}",
            })
        if TEACHER_CKPT.is_file():
            models.append({
                "id": "teacher_s8000",
                "name": "Teacher model_s8000 (RF-40 天花板基准)",
                "flow": "rf",
                "default_steps": 40,
                "path": str(TEACHER_CKPT),
            })
        return models

    def _ckpt_path(self, model_id: str) -> Path:
        if model_id == "clarity_fix":
            return CLARITY_CKPT
        if model_id == "mf_20k":
            return MF20K_CKPT
        if model_id == "teacher_s8000":
            return TEACHER_CKPT
        raise ValueError(f"Unknown checkpoint model_id: {model_id}")

    def _resolve_device(self, device: str) -> str:
        d = str(device or "cpu").strip().lower()
        if d == "auto_gpu":
            return "cuda" if self._cuda_available else "cpu"
        if d == "cuda" and not self._cuda_available:
            return "cpu"
        if d not in ("cpu", "cuda"):
            return "cpu"
        return d

    def _get_runtime_locked(self, model_id: str, device: str) -> InferenceRuntime:
        dev = self._resolve_device(device)
        cache_key = (model_id, dev)
        if cache_key in self._runtimes:
            return self._runtimes[cache_key]

        ckpt = self._ckpt_path(model_id)
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        precision = "bf16" if dev == "cuda" else "fp32"
        t0 = time.perf_counter()
        print(f"[workbench] loading {model_id} on {dev} ({precision})...", flush=True)
        rt = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(ckpt),
                model_device=dev,
                codec_repo=CODEC_REPO,
                model_precision=precision,
                codec_device=dev,
                codec_precision="fp32",
                compile_model=False,
            )
        )
        elapsed = time.perf_counter() - t0
        print(f"[workbench] loaded {model_id} on {dev} in {elapsed:.2f}s", flush=True)
        self._runtimes[cache_key] = rt
        return rt

    def preload_default_cpu(self) -> None:
        self._preload_state = "loading"
        t0 = time.perf_counter()
        try:
            with self._lock:
                self._get_runtime_locked("clarity_fix", "cpu")
            self._preload_sec = round(time.perf_counter() - t0, 2)
            self._preload_state = "ready"
        except Exception as exc:
            self._preload_error = str(exc)
            self._preload_state = "error"
            print(f"[workbench] preload failed: {exc}", flush=True)

    def status_dict(self) -> dict[str, Any]:
        return {
            "preload_state": self._preload_state,
            "preload_sec": self._preload_sec,
            "preload_error": self._preload_error,
            "loaded_runtimes": [f"{m}@{d}" for (m, d) in sorted(self._runtimes.keys())],
            "cuda_available": self._cuda_available,
            "cuda_name": self._cuda_name,
            "cpu_count": os.cpu_count() or 1,
            "torch_threads": torch.get_num_threads(),
            "default_threads": self._default_threads,
            "models": self.available_models(),
            "speakers": [
                {
                    **sp,
                    "embed_ready": (SPEAKERS_DIR / f"{sp['id']}.speaker.safetensors").is_file(),
                }
                for sp in SPEAKER_CATALOG
            ],
            "caption_chips": [{"label": lbl, "caption": cap} for lbl, cap in CAPTION_CHIPS],
            "bench_columns": BENCH_COLUMNS,
            "bench_job": dict(self._bench_job),
        }

    def _synthesize_hybrid_locked(
        self,
        *,
        mf_model_id: str,
        device: str,
        req: SamplingRequest,
        mf_steps: int = 3,
        rf_steps: int = 5,
        split_t: float = 0.25,
    ) -> dict[str, Any]:
        """Run MeanFlow from 1.0 -> split_t, then Teacher RF from split_t -> 0.0 (no CFG)."""
        dev = self._resolve_device(device)
        mf_rt = self._get_runtime_locked(mf_model_id, dev)
        rf_rt = self._get_runtime_locked("teacher_s8000", dev)

        noop = lambda _msg: None
        post_load_t0 = time.perf_counter()
        plan = mf_rt._plan_request(req, log_fn=noop)
        (
            text_ids,
            text_mask,
            caption_ids,
            caption_mask,
            speaker_state_override,
            speaker_mask_override,
            ref_latent,
            ref_mask,
        ) = mf_rt._prepare_request_conditions(plan, log_fn=noop)

        t_samp = time.perf_counter()
        batch_size = text_ids.shape[0]
        mf_model = mf_rt.model
        rf_model = rf_rt.model
        mdev = mf_model.device
        mdtype = mf_model.dtype

        with torch.inference_mode():
            x_t = draw_initial_noise(
                plan.used_seed,
                (batch_size, plan.patched_steps, mf_model.cfg.patched_latent_dim),
                device=mdev,
                dtype=mdtype,
            )[0]

            # 1) MeanFlow front stage: 1.0 -> split_t
            mf_encoded = EncodedConditions(
                *mf_model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                )
            )
            mf_kv = mf_model.build_context_kv_cache(
                text_state=mf_encoded.text_state,
                speaker_state=mf_encoded.speaker_state,
                caption_state=mf_encoded.caption_state,
            )
            mf_sched = torch.linspace(1.0, float(split_t), mf_steps + 1, device=mdev, dtype=torch.float32)
            for idx in range(mf_steps):
                t_val = mf_sched[idx]
                next_val = mf_sched[idx + 1]
                t_vec = t_val.expand(batch_size)
                delta = (t_val - next_val).expand(batch_size)
                vel = mf_model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=t_vec,
                    delta_t=delta,
                    text_state=mf_encoded.text_state,
                    text_mask=mf_encoded.text_mask,
                    speaker_state=mf_encoded.speaker_state,
                    speaker_mask=mf_encoded.speaker_mask,
                    caption_state=mf_encoded.caption_state,
                    caption_mask=mf_encoded.caption_mask,
                    context_kv_cache=mf_kv,
                )
                x_t = x_t + vel * (next_val - t_val)

            t_mf_sec = time.perf_counter() - t_samp
            plan.stage_timings.append(("sample_mf_front", t_mf_sec))

            # 2) Teacher RF tail stage: split_t -> 0.0 (t <= 0.25 < cfg_min_t=0.5, single branch no CFG)
            t_rf0 = time.perf_counter()
            rf_encoded = EncodedConditions(
                *rf_model.encode_conditions(
                    text_input_ids=text_ids,
                    text_mask=text_mask,
                    ref_latent=ref_latent,
                    ref_mask=ref_mask,
                    caption_input_ids=caption_ids,
                    caption_mask=caption_mask,
                    speaker_state_override=speaker_state_override,
                    speaker_mask_override=speaker_mask_override,
                    speaker_uncond_mode=req.speaker_uncond_mode,
                )
            )
            rf_kv = rf_model.build_context_kv_cache(
                text_state=rf_encoded.text_state,
                speaker_state=rf_encoded.speaker_state,
                caption_state=rf_encoded.caption_state,
            )
            rf_sched = torch.linspace(float(split_t), 0.0, rf_steps + 1, device=mdev, dtype=torch.float32)
            for idx in range(rf_steps):
                t_val = rf_sched[idx]
                next_val = rf_sched[idx + 1]
                t_vec = t_val.expand(batch_size)
                vel = rf_model.forward_with_encoded_conditions(
                    x_t=x_t,
                    t=t_vec,
                    text_state=rf_encoded.text_state,
                    text_mask=rf_encoded.text_mask,
                    speaker_state=rf_encoded.speaker_state,
                    speaker_mask=rf_encoded.speaker_mask,
                    caption_state=rf_encoded.caption_state,
                    caption_mask=rf_encoded.caption_mask,
                    context_kv_cache=rf_kv,
                )
                x_t = x_t + vel * (next_val - t_val)
            plan.stage_timings.append(("sample_rf_tail", time.perf_counter() - t_rf0))

        result = mf_rt._finish_request(plan, x_t, post_load_t0=post_load_t0, log_fn=noop)
        return self._pack_result(result, dev)

    @staticmethod
    def _pack_result(result: Any, actual_device: str) -> dict[str, Any]:
        audio = result.audio
        sr = int(result.sample_rate)
        audio_sec = float(audio.shape[-1]) / float(sr)
        total_sec = float(result.total_to_decode)
        rtf = total_sec / max(1e-6, audio_sec)
        return {
            "audio": audio,
            "sample_rate": sr,
            "audio_sec": round(audio_sec, 3),
            "total_sec": round(total_sec, 3),
            "rtf": round(rtf, 3),
            "used_seed": int(result.used_seed),
            "device": actual_device,
            "stage_timings": [
                {"name": name, "ms": round(sec * 1000.0, 1)}
                for name, sec in result.stage_timings
            ],
        }

    def _synthesize_single_locked(
        self,
        *,
        embed_path: Path,
        text_clean: str,
        caption_clean: str | None,
        model_id: str,
        dev: str,
        num_steps: int,
        used_seed: int,
        duration_scale: float,
    ) -> dict[str, Any]:
        if model_id in ("hybrid_clarity_teacher", "hybrid_20k_teacher"):
            mf_id = "clarity_fix" if model_id == "hybrid_clarity_teacher" else "mf_20k"
            req = SamplingRequest(
                text=text_clean,
                caption=caption_clean,
                ref_embed=str(embed_path),
                num_steps=4,
                seed=used_seed,
                duration_scale=float(duration_scale),
            )
            out = self._synthesize_hybrid_locked(
                mf_model_id=mf_id,
                device=dev,
                req=req,
                mf_steps=3,
                rf_steps=5,
                split_t=0.25,
            )
            out["num_steps"] = 8
            out["threads"] = torch.get_num_threads() if dev == "cpu" else None
            return out

        rt = self._get_runtime_locked(model_id, dev)
        is_rf = str(rt.model_cfg.flow_parameterization).strip().lower() != "meanflow"
        steps = int(num_steps) if num_steps else (40 if is_rf else 4)
        req_kwargs: dict[str, Any] = dict(
            text=text_clean,
            caption=caption_clean,
            ref_embed=str(embed_path),
            num_steps=steps,
            seed=used_seed,
            duration_scale=float(duration_scale),
        )
        if is_rf:
            req_kwargs.update(
                cfg_guidance_mode="independent",
                cfg_scale_text=3.0,
                cfg_scale_caption=4.0,
                cfg_scale_speaker=5.0,
            )
        res = rt.synthesize(SamplingRequest(**req_kwargs), log_fn=None)
        out = self._pack_result(res, dev)
        out["num_steps"] = steps
        out["threads"] = torch.get_num_threads() if dev == "cpu" else None
        return out

    def synthesize(
        self,
        *,
        speaker_id: str,
        text: str,
        caption: str = "",
        model_id: str = "clarity_fix",
        device: str = "cpu",
        num_steps: int = 4,
        seed: int | None = 42,
        duration_scale: float = 1.0,
        cpu_threads: int | None = None,
        auto_chunk: bool = False,
        max_chars: int = 45,
        pause_ms: int = 220,
    ) -> dict[str, Any]:
        embed_path = SPEAKERS_DIR / f"{speaker_id}.speaker.safetensors"
        if not embed_path.is_file():
            raise FileNotFoundError(f"Speaker embedding not found: {embed_path}")
        text_clean = str(text or "").strip()
        if not text_clean:
            raise ValueError("请输入要合成的日文台词文本。")
        caption_clean = str(caption or "").strip() or None
        dev = self._resolve_device(device)

        if seed is None or str(seed).strip() == "":
            used_seed = int(secrets.randbelow(2**31 - 1))
        else:
            used_seed = int(seed)

        chunks = (
            split_japanese_chunks(text_clean, max_chars=int(max_chars))
            if auto_chunk
            else [text_clean]
        )

        with self._lock:
            if cpu_threads is not None and dev == "cpu":
                t_clamped = max(1, min(os.cpu_count() or 16, int(cpu_threads)))
                if torch.get_num_threads() != t_clamped:
                    torch.set_num_threads(t_clamped)

            if len(chunks) <= 1:
                out = self._synthesize_single_locked(
                    embed_path=embed_path,
                    text_clean=text_clean,
                    caption_clean=caption_clean,
                    model_id=model_id,
                    dev=dev,
                    num_steps=num_steps,
                    used_seed=used_seed,
                    duration_scale=duration_scale,
                )
                out["chunked"] = False
                out["chunk_count"] = 1
                out["ttfb_sec"] = out["total_sec"]
                out["chunks"] = []
                return out

            audios: list[torch.Tensor] = []
            chunk_infos: list[dict[str, Any]] = []
            stage_acc: dict[str, float] = {}
            stage_order: list[str] = []
            total_infer = 0.0
            sample_rate = 48000
            actual_steps = num_steps
            actual_threads = None

            for idx, chunk_text in enumerate(chunks):
                sub = self._synthesize_single_locked(
                    embed_path=embed_path,
                    text_clean=chunk_text,
                    caption_clean=caption_clean,
                    model_id=model_id,
                    dev=dev,
                    num_steps=num_steps,
                    used_seed=used_seed + idx,
                    duration_scale=duration_scale,
                )
                audios.append(sub["audio"])
                sample_rate = sub["sample_rate"]
                actual_steps = sub["num_steps"]
                actual_threads = sub["threads"]
                total_infer += float(sub["total_sec"])
                chunk_infos.append({
                    "index": idx + 1,
                    "text": chunk_text,
                    "audio_sec": sub["audio_sec"],
                    "total_sec": sub["total_sec"],
                    "rtf": sub["rtf"],
                })
                for st in sub["stage_timings"]:
                    name = st["name"]
                    if name not in stage_acc:
                        stage_acc[name] = 0.0
                        stage_order.append(name)
                    stage_acc[name] += float(st["ms"])

            stitched = stitch_audio_chunks(
                audios,
                sample_rate,
                pause_ms=int(pause_ms),
                chunk_texts=chunks,
            )
            audio_sec = float(stitched.shape[-1]) / float(sample_rate)
            rtf = total_infer / max(1e-6, audio_sec)
            return {
                "audio": stitched,
                "sample_rate": sample_rate,
                "audio_sec": round(audio_sec, 3),
                "total_sec": round(total_infer, 3),
                "rtf": round(rtf, 3),
                "used_seed": used_seed,
                "device": dev,
                "num_steps": actual_steps,
                "threads": actual_threads,
                "stage_timings": [
                    {"name": name, "ms": round(stage_acc[name], 1)}
                    for name in stage_order
                ],
                "chunked": True,
                "chunk_count": len(chunks),
                "ttfb_sec": chunk_infos[0]["total_sec"],
                "chunks": chunk_infos,
            }

    def bench_cell_paths(self, case_id: str, col_id: str) -> tuple[Path, Path]:
        d = BENCH_DIR / case_id
        return d / f"{col_id}.wav", d / f"{col_id}.json"

    def get_bench_state(self) -> dict[str, Any]:
        cases_out: list[dict[str, Any]] = []
        for case in BENCH_CASES:
            cells: dict[str, Any] = {}
            for col in BENCH_COLUMNS:
                wav_p, meta_p = self.bench_cell_paths(case["id"], col["id"])
                if wav_p.is_file() and meta_p.is_file():
                    try:
                        meta = json.loads(meta_p.read_text(encoding="utf-8"))
                        meta["audio_url"] = f"/api/bench_audio/{case['id']}/{col['id']}.wav?v={int(wav_p.stat().st_mtime)}"
                        meta["ready"] = True
                        cells[col["id"]] = meta
                    except Exception:
                        cells[col["id"]] = {"ready": False}
                else:
                    cells[col["id"]] = {"ready": False}
            sp = SPEAKER_BY_ID.get(case["speaker_id"], {"ja": case["speaker_id"], "zh": ""})
            cases_out.append({
                **case,
                "speaker_ja": sp["ja"],
                "speaker_zh": sp["zh"],
                "cells": cells,
            })
        return {
            "columns": BENCH_COLUMNS,
            "cases": cases_out,
            "job": dict(self._bench_job),
        }

    def run_bench_cell(self, case_id: str, col_id: str, *, force: bool = False) -> dict[str, Any]:
        case = BENCH_CASE_BY_ID[case_id]
        col = BENCH_COL_BY_ID[col_id]
        wav_p, meta_p = self.bench_cell_paths(case_id, col_id)
        if not force and wav_p.is_file() and meta_p.is_file():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            meta["audio_url"] = f"/api/bench_audio/{case_id}/{col_id}.wav?v={int(wav_p.stat().st_mtime)}"
            meta["ready"] = True
            return meta

        out = self.synthesize(
            speaker_id=case["speaker_id"],
            text=case["text"],
            caption=case["caption"],
            model_id=col["model"],
            device=col["device"],
            num_steps=int(col["num_steps"]),
            seed=int(case.get("seed", 42)),
            duration_scale=1.0,
            cpu_threads=self._default_threads,
            auto_chunk=bool(col.get("auto_chunk", False)),
            max_chars=int(col.get("max_chars", 45)),
            pause_ms=int(col.get("pause_ms", 220)),
        )
        wav_p.parent.mkdir(parents=True, exist_ok=True)
        save_wav(wav_p, out["audio"], out["sample_rate"])
        meta = {
            "case_id": case_id,
            "col_id": col_id,
            "model": col["model"],
            "device": out["device"],
            "num_steps": out["num_steps"],
            "threads": out["threads"],
            "audio_sec": out["audio_sec"],
            "total_sec": out["total_sec"],
            "rtf": out["rtf"],
            "used_seed": out["used_seed"],
            "sample_rate": out["sample_rate"],
            "stage_timings": out["stage_timings"],
            "chunked": out.get("chunked", False),
            "chunk_count": out.get("chunk_count", 1),
            "ttfb_sec": out.get("ttfb_sec", out["total_sec"]),
            "chunks": out.get("chunks", []),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta_p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        meta["audio_url"] = f"/api/bench_audio/{case_id}/{col_id}.wav?v={int(wav_p.stat().st_mtime)}"
        meta["ready"] = True
        return meta

    def start_bench_batch(
        self,
        *,
        case_ids: list[str] | None = None,
        col_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        with self._bench_job_lock:
            if self._bench_job["running"]:
                return {"started": False, "reason": "already_running", "job": dict(self._bench_job)}

            c_ids = case_ids or [c["id"] for c in BENCH_CASES]
            k_ids = col_ids or [c["id"] for c in BENCH_COLUMNS]
            tasks: list[tuple[str, str]] = []
            # Group by col_id so each model stays hot in cache
            for col_id in k_ids:
                if col_id not in BENCH_COL_BY_ID:
                    continue
                for case_id in c_ids:
                    if case_id not in BENCH_CASE_BY_ID:
                        continue
                    wav_p, meta_p = self.bench_cell_paths(case_id, col_id)
                    if force or not (wav_p.is_file() and meta_p.is_file()):
                        tasks.append((case_id, col_id))

            if not tasks:
                return {"started": False, "reason": "nothing_to_run", "job": dict(self._bench_job)}

            self._bench_job = {
                "running": True,
                "done": 0,
                "total": len(tasks),
                "current": f"{tasks[0][0]} / {tasks[0][1]}",
                "error": None,
            }

            def _worker() -> None:
                try:
                    for idx, (cid, kid) in enumerate(tasks):
                        with self._bench_job_lock:
                            self._bench_job["current"] = f"{cid} · {BENCH_COL_BY_ID[kid]['short']}"
                        self.run_bench_cell(cid, kid, force=force)
                        with self._bench_job_lock:
                            self._bench_job["done"] = idx + 1
                except Exception as exc:
                    with self._bench_job_lock:
                        self._bench_job["error"] = str(exc)
                    print(f"[workbench] bench batch error: {exc}", flush=True)
                finally:
                    with self._bench_job_lock:
                        self._bench_job["running"] = False
                        self._bench_job["current"] = ""

            threading.Thread(target=_worker, daemon=True).start()
            return {"started": True, "job": dict(self._bench_job)}


_history_lock = threading.Lock()


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_INDEX.is_file():
        return []
    try:
        items = json.loads(HISTORY_INDEX.read_text(encoding="utf-8"))
        return items if isinstance(items, list) else []
    except Exception:
        return []


def append_history(entry: dict[str, Any]) -> None:
    with _history_lock:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        items = load_history()
        items.insert(0, entry)
        items = items[:80]
        HISTORY_INDEX.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>SC29 MeanFlow CPU 推理工作台 (clarity_fix)</title>
<style>
:root {
  --bg: #0c1018;
  --panel: #141a26;
  --panel-2: #1a2233;
  --border: #263147;
  --text: #e6ecf8;
  --muted: #8b9bb8;
  --accent: #5b9dff;
  --accent-2: #34d399;
  --warn: #fbbf24;
  --purple: #a78bfa;
  --danger: #f87171;
  --radius: 12px;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", "Hiragino Sans", sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}
header {
  position: sticky; top: 0; z-index: 20;
  background: rgba(12, 16, 24, 0.92);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
  padding: 12px 24px;
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
}
.brand { display: flex; align-items: center; gap: 12px; }
.brand h1 { margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0.2px; }
.pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
}
.pill.ok { color: var(--accent-2); border-color: rgba(52, 211, 153, 0.35); background: rgba(52, 211, 153, 0.08); }
.pill.loading { color: var(--warn); border-color: rgba(251, 191, 36, 0.35); background: rgba(251, 191, 36, 0.08); }
.pill.cpu { color: #7dd3fc; border-color: rgba(125, 211, 252, 0.35); background: rgba(125, 211, 252, 0.08); }
.tabs { display: flex; gap: 8px; }
.tab-btn {
  padding: 8px 16px; border-radius: 8px; border: 1px solid var(--border);
  background: var(--panel); color: var(--muted); cursor: pointer; font-weight: 600; font-size: 13px;
  transition: all 0.15s;
}
.tab-btn.active { background: var(--accent); color: #071224; border-color: var(--accent); }
.tab-btn:hover:not(.active) { border-color: var(--accent); color: var(--text); }
main { max-width: 1480px; margin: 0 auto; padding: 20px 24px 60px; }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

.studio-grid {
  display: grid; grid-template-columns: minmax(380px, 520px) 1fr; gap: 20px; align-items: start;
}
@media (max-width: 1020px) { .studio-grid { grid-template-columns: 1fr; } }

.card {
  background: var(--panel); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 18px;
}
.card h2 { margin: 0 0 14px; font-size: 15px; font-weight: 700; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.field { margin-bottom: 14px; }
.field label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 6px; font-weight: 600; }
input[type="text"], input[type="number"], textarea, select {
  width: 100%; background: var(--panel-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 8px; padding: 9px 11px;
  font-size: 13.5px; font-family: inherit; outline: none;
}
input:focus, textarea:focus, select:focus { border-color: var(--accent); }
textarea { resize: vertical; min-height: 76px; }
.row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.row-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }

.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.chip {
  font-size: 11.5px; padding: 4px 9px; border-radius: 999px;
  background: var(--panel-2); border: 1px solid var(--border); color: var(--muted);
  cursor: pointer; transition: all 0.12s;
}
.chip:hover { border-color: var(--accent); color: var(--text); }

.btn-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
.btn {
  padding: 10px 16px; border-radius: 9px; border: 1px solid var(--border);
  background: var(--panel-2); color: var(--text); font-weight: 600; font-size: 13.5px;
  cursor: pointer; transition: all 0.15s; display: inline-flex; align-items: center; gap: 7px;
}
.btn.primary { background: var(--accent); color: #061326; border-color: var(--accent); }
.btn.primary:hover { filter: brightness(1.08); }
.btn.secondary { background: rgba(167, 139, 250, 0.14); border-color: rgba(167, 139, 250, 0.45); color: #ddd6fe; }
.btn.secondary:hover { background: rgba(167, 139, 250, 0.24); }
.btn.sm { padding: 5px 10px; font-size: 12px; border-radius: 7px; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }

.result-item {
  background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px; margin-bottom: 12px;
}
.result-item.fresh { border-color: rgba(91, 157, 255, 0.55); }
.result-head { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; }
.result-title { font-weight: 700; font-size: 14px; }
.metrics { display: flex; flex-wrap: wrap; gap: 6px; font-size: 12px; }
.metric {
  padding: 2px 8px; border-radius: 6px; background: rgba(255,255,255,0.04);
  border: 1px solid var(--border); color: var(--muted);
}
.metric strong { color: var(--text); }
.metric.rtf-fast strong { color: var(--accent-2); }
.metric.rtf-mid strong { color: var(--warn); }
.result-text { font-size: 13px; color: var(--text); margin: 4px 0; }
.result-cap { font-size: 12px; color: var(--muted); margin: 2px 0 10px; }
audio { width: 100%; height: 36px; border-radius: 6px; }
.stage-bar {
  display: flex; height: 7px; border-radius: 99px; overflow: hidden;
  background: #0d121c; margin-top: 8px;
}
.stage-legend { display: flex; flex-wrap: wrap; gap: 10px; font-size: 11px; color: var(--muted); margin-top: 5px; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; vertical-align: middle; }

/* Benchmark table */
.bench-toolbar {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px;
  margin-bottom: 14px;
}
.bench-table-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: var(--radius); background: var(--panel); }
table.bench { width: 100%; border-collapse: collapse; min-width: 1080px; }
table.bench th, table.bench td {
  border-bottom: 1px solid var(--border); border-right: 1px solid rgba(38, 49, 71, 0.55);
  padding: 12px; vertical-align: top; text-align: left;
}
table.bench th { background: var(--panel-2); font-size: 12.5px; color: var(--text); position: sticky; top: 0; }
table.bench td.prompt-col { width: 320px; min-width: 280px; }
.case-title { font-weight: 700; font-size: 13.5px; margin-bottom: 4px; }
.case-tag {
  display: inline-block; font-size: 11px; padding: 1px 7px; border-radius: 999px;
  background: rgba(91, 157, 255, 0.14); color: #93c5fd; margin-bottom: 6px;
}
.case-text { font-size: 12.5px; margin-bottom: 4px; }
.case-cap { font-size: 11.5px; color: var(--muted); margin-bottom: 8px; }
.cell-box { display: flex; flex-direction: column; gap: 6px; min-width: 215px; }
.cell-meta { font-size: 11.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px; }
.cell-empty {
  border: 1px dashed var(--border); border-radius: 8px; padding: 14px 10px;
  text-align: center; color: var(--muted); font-size: 12px;
}
.progress-bar-wrap {
  height: 6px; width: 220px; background: var(--panel-2); border-radius: 99px; overflow: hidden;
  border: 1px solid var(--border);
}
.progress-bar-fill { height: 100%; background: var(--accent-2); width: 0%; transition: width 0.25s; }
.compare-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 10px; margin-top: 10px; }
.compare-card {
  background: #101622; border: 1px solid var(--border); border-radius: 8px; padding: 10px;
}
.error-banner {
  background: rgba(248, 113, 113, 0.12); border: 1px solid rgba(248, 113, 113, 0.4);
  color: #fecaca; padding: 10px 14px; border-radius: 8px; font-size: 13px; margin-bottom: 12px;
  display: none;
}
.muted { color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
<header>
  <div class="brand">
    <h1>SC29 MeanFlow CPU 推理工作台</h1>
    <span id="status-pill" class="pill loading">正在加载 CPU 模型…</span>
    <span id="hw-pill" class="pill cpu">CPU</span>
  </div>
  <div class="tabs">
    <button class="tab-btn active" data-tab="bench" onclick="switchTab('bench')">验收对照台 (CPU 推理结果)</button>
    <button class="tab-btn" data-tab="studio" onclick="switchTab('studio')">实时 CPU 推理工作台 (29 角色)</button>
  </div>
</header>

<main>
  <div id="global-err" class="error-banner"></div>

  <!-- TAB 1: BENCHMARK COMPARISON MATRIX -->
  <section id="tab-bench" class="tab-pane active">
    <div class="bench-toolbar">
      <div>
        <div style="font-size:15px;font-weight:700;">clarity_fix_1500 CPU 推理验收矩阵（29 角色 49 KB --ref-embed）</div>
        <div class="muted">同种子、同 49 KB 角色 Embed 直接对比：clarity_fix 4步 (CPU) · clarity_fix 8步 (CPU) · 混合 MF3+RF5 (CPU) · Teacher s8000 RF-40 天花板。所有结果本地持久化。</div>
      </div>
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
        <div id="bench-progress-box" style="display:none;align-items:center;gap:8px;">
          <span id="bench-progress-txt" class="muted"></span>
          <div class="progress-bar-wrap"><div id="bench-progress-fill" class="progress-bar-fill"></div></div>
        </div>
        <button class="btn primary sm" onclick="runBenchBatch(['clarity_mf4_chunk_cpu','clarity_mf8_chunk_cpu'])">补齐 分句拼接 (4步+8步 CPU)</button>
        <button class="btn sm" onclick="runBenchBatch(['clarity_mf4_cpu','clarity_mf8_cpu','teacher_rf40'])">补齐 4步/8步/Teacher</button>
        <button class="btn secondary sm" onclick="runBenchBatch(['hybrid_3_5_cpu'])">补齐 混合 MF3+RF5 (CPU)</button>
        <button class="btn sm" onclick="refreshBench()">刷新</button>
      </div>
    </div>
    <div class="bench-table-wrap">
      <table class="bench">
        <thead>
          <tr id="bench-thead"></tr>
        </thead>
        <tbody id="bench-tbody"></tbody>
      </table>
    </div>
  </section>

  <!-- TAB 2: INTERACTIVE STUDIO -->
  <section id="tab-studio" class="tab-pane">
    <div class="studio-grid">
      <div class="card">
        <h2>
          <span>合成参数（默认纯 CPU 推理）</span>
          <span class="muted">全部使用 49 KB --ref-embed</span>
        </h2>

        <div class="field">
          <label>快捷诊断预设（点击填入）</label>
          <div id="preset-chips" class="chips"></div>
        </div>

        <div class="field">
          <label for="speaker-select">角色（全量 29 位 SC 角色）</label>
          <select id="speaker-select"></select>
        </div>

        <div class="field">
          <label for="text-input">台词文本（日文）</label>
          <textarea id="text-input">プロデューサーさん、今日のお仕事もふゆにお任せくださいねっ！</textarea>
        </div>

        <div class="field">
          <label for="caption-input">表演风格 / 情绪 Caption（可选）</label>
          <textarea id="caption-input" style="min-height:60px;">猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。</textarea>
          <div id="caption-chips" class="chips"></div>
        </div>

        <div class="field">
          <label for="model-select">模型 / 推理模式</label>
          <select id="model-select" onchange="onModelChange()"></select>
        </div>

        <div class="row-3">
          <div class="field">
            <label for="device-select">推理设备</label>
            <select id="device-select">
              <option value="cpu" selected>CPU (FP32)</option>
              <option value="cuda">GPU (RTX 4090)</option>
            </select>
          </div>
          <div class="field">
            <label for="steps-select">采样步数 (NFE)</label>
            <select id="steps-select">
              <option value="4" selected>4 步 (MeanFlow 默认)</option>
              <option value="6">6 步</option>
              <option value="8">8 步</option>
              <option value="12">12 步</option>
              <option value="16">16 步</option>
              <option value="40">40 步 (Teacher RF)</option>
            </select>
          </div>
          <div class="field">
            <label for="threads-select">CPU 线程数</label>
            <select id="threads-select">
              <option value="4">4 线程</option>
              <option value="8">8 线程 (模拟 8 核 Xeon)</option>
              <option value="10" selected>10 线程 (本机默认)</option>
              <option value="16">16 线程</option>
            </select>
          </div>
        </div>

        <div class="row-2">
          <div class="field">
            <label for="seed-input">随机种子（留空随机）</label>
            <input id="seed-input" type="text" value="42" placeholder="42 或留空随机" />
          </div>
          <div class="field">
            <label for="dur-scale">语速 / 时长倍率 (duration_scale)</label>
            <input id="dur-scale" type="number" step="0.05" min="0.6" max="1.6" value="1.0" />
          </div>
        </div>

        <div class="field" style="background:var(--panel-2);border:1px solid var(--border);border-radius:10px;padding:12px;">
          <label style="display:flex;align-items:center;gap:8px;cursor:pointer;margin-bottom:8px;color:var(--text);font-size:13px;">
            <input id="auto-chunk" type="checkbox" checked onchange="toggleChunkOptions()" style="width:16px;height:16px;accent-color:var(--accent-2);" />
            <span>自动按标点分句拼接（长语音防漂移 + 降低首句延迟 TTFB）</span>
          </label>
          <div id="chunk-opts" class="row-2">
            <div>
              <label for="max-chars" style="margin-bottom:4px;">单段上限字数（约 4–9s 甜蜜区）</label>
              <input id="max-chars" type="number" min="20" max="80" step="5" value="45" />
            </div>
            <div>
              <label for="pause-ms" style="margin-bottom:4px;">句间自然停顿 (ms)</label>
              <input id="pause-ms" type="number" min="0" max="600" step="20" value="220" />
            </div>
          </div>
        </div>

        <div class="btn-row">
          <button id="btn-gen" class="btn primary" onclick="runSynthesize()">开始推理生成</button>
          <button id="btn-cmp" class="btn secondary" onclick="runCompare()">一键同台对比（含整段 vs 分句）</button>
        </div>
        <div id="studio-status" class="muted" style="margin-top:10px;"></div>
      </div>

      <div class="card">
        <h2>
          <span>推理结果与历史记录</span>
          <span id="hist-count" class="muted"></span>
        </h2>
        <div id="compare-area"></div>
        <div id="history-list">
          <div class="muted">暂无实时合成记录。点击左侧“开始推理生成”或“一键三路同台对比”即可试听。</div>
        </div>
      </div>
    </div>
  </section>
</main>

<script>
let STATE = null;
let BENCH = null;
let benchPollTimer = null;

function showErr(msg) {
  const el = document.getElementById('global-err');
  if (!msg) { el.style.display = 'none'; el.textContent = ''; return; }
  el.style.display = 'block';
  el.textContent = msg;
}

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === 'tab-' + name));
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function stageColors(name) {
  if (name.includes('duration')) return '#60a5fa';
  if (name.includes('sample_mf') || name.includes('meanflow')) return '#34d399';
  if (name.includes('sample_rf') || name.includes('rf')) return '#a78bfa';
  if (name.includes('decode')) return '#fbbf24';
  return '#94a3b8';
}

function renderStageBar(timings) {
  if (!timings || !timings.length) return '';
  const kept = timings.filter(t => t.ms >= 1.0);
  const sum = kept.reduce((a, b) => a + b.ms, 0) || 1;
  const segs = kept.map(t => `<div style="width:${(t.ms/sum*100).toFixed(1)}%;background:${stageColors(t.name)}" title="${t.name}: ${t.ms} ms"></div>`).join('');
  const legend = kept.map(t => `<span><i class="dot" style="background:${stageColors(t.name)}"></i>${escapeHtml(t.name)} ${t.ms}ms</span>`).join('');
  return `<div class="stage-bar">${segs}</div><div class="stage-legend">${legend}</div>`;
}

async function fetchStatus() {
  const res = await fetch('/api/status');
  STATE = await res.json();
  const pill = document.getElementById('status-pill');
  if (STATE.preload_state === 'ready') {
    pill.className = 'pill ok';
    pill.textContent = `CPU 模型已就绪 (${STATE.preload_sec ?? ''}s) · 已驻留: ${STATE.loaded_runtimes.join(', ')}`;
  } else if (STATE.preload_state === 'error') {
    pill.className = 'pill';
    pill.textContent = '加载失败: ' + STATE.preload_error;
  } else {
    pill.className = 'pill loading';
    pill.textContent = '正在预热 clarity_fix CPU 模型…';
    setTimeout(fetchStatus, 1200);
  }
  const hw = document.getElementById('hw-pill');
  hw.textContent = `CPU ${STATE.torch_threads}/${STATE.cpu_count} 线程` + (STATE.cuda_available ? ` · GPU: ${STATE.cuda_name}` : '');
  populateControls();
}

let controlsPopulated = false;
function populateControls() {
  if (controlsPopulated || !STATE) return;
  controlsPopulated = true;

  // Speakers grouped by unit
  const spSel = document.getElementById('speaker-select');
  const units = {};
  for (const sp of STATE.speakers) {
    (units[sp.unit] = units[sp.unit] || []).push(sp);
  }
  spSel.innerHTML = Object.entries(units).map(([unit, list]) => {
    const opts = list.map(sp =>
      `<option value="${sp.id}" ${sp.id === 'fuyuko' ? 'selected' : ''}>${escapeHtml(sp.ja)} / ${escapeHtml(sp.zh)} (${sp.id})${sp.embed_ready ? '' : ' [缺失]'}</option>`
    ).join('');
    return `<optgroup label="${escapeHtml(unit)}">${opts}</optgroup>`;
  }).join('');

  // Models
  const mSel = document.getElementById('model-select');
  mSel.innerHTML = STATE.models.map(m =>
    `<option value="${m.id}">${escapeHtml(m.name)}</option>`
  ).join('');

  // Caption chips
  const capChips = document.getElementById('caption-chips');
  capChips.innerHTML = STATE.caption_chips.map((c, idx) =>
    `<span class="chip" onclick="applyCaption(${idx})">${escapeHtml(c.label)}</span>`
  ).join('');

  // Disable CUDA if unavailable
  if (!STATE.cuda_available) {
    const devSel = document.getElementById('device-select');
    const opt = devSel.querySelector('option[value="cuda"]');
    if (opt) opt.disabled = true;
  }
}

function applyCaption(idx) {
  document.getElementById('caption-input').value = STATE.caption_chips[idx].caption;
}

function onModelChange() {
  const val = document.getElementById('model-select').value;
  const stepsSel = document.getElementById('steps-select');
  const devSel = document.getElementById('device-select');
  if (val === 'teacher_s8000') {
    stepsSel.value = '40';
    if (STATE && STATE.cuda_available) devSel.value = 'cuda';
  } else if (val.startsWith('hybrid_')) {
    stepsSel.value = '8';
    devSel.value = 'cpu';
  } else {
    stepsSel.value = '4';
    devSel.value = 'cpu';
  }
}

function loadPresetToStudio(caseId) {
  if (!BENCH) return;
  const c = BENCH.cases.find(x => x.id === caseId);
  if (!c) return;
  document.getElementById('speaker-select').value = c.speaker_id;
  document.getElementById('text-input').value = c.text;
  document.getElementById('caption-input').value = c.caption;
  document.getElementById('seed-input').value = String(c.seed ?? 42);
  switchTab('studio');
}

function populatePresetChips() {
  if (!BENCH) return;
  const box = document.getElementById('preset-chips');
  box.innerHTML = BENCH.cases.map(c =>
    `<span class="chip" onclick="loadPresetToStudio('${c.id}')">${escapeHtml(c.title)}</span>`
  ).join('');
}

async function refreshBench() {
  const res = await fetch('/api/bench/status');
  BENCH = await res.json();
  populatePresetChips();
  renderBenchTable();
  updateBenchProgress(BENCH.job);
}

function updateBenchProgress(job) {
  const box = document.getElementById('bench-progress-box');
  const txt = document.getElementById('bench-progress-txt');
  const fill = document.getElementById('bench-progress-fill');
  if (job && job.running) {
    box.style.display = 'flex';
    const pct = job.total ? Math.round((job.done / job.total) * 100) : 0;
    txt.textContent = `生成中 ${job.done}/${job.total} · ${job.current}`;
    fill.style.width = pct + '%';
    if (!benchPollTimer) {
      benchPollTimer = setInterval(refreshBench, 1800);
    }
  } else {
    box.style.display = 'none';
    if (benchPollTimer) { clearInterval(benchPollTimer); benchPollTimer = null; }
    if (job && job.error) showErr('批量生成出错: ' + job.error);
  }
}

function renderBenchTable() {
  if (!BENCH) return;
  const thead = document.getElementById('bench-thead');
  thead.innerHTML = `<th>测试样例 / 角色 / 台词</th>` + BENCH.columns.map(col =>
    `<th>${escapeHtml(col.label)}</th>`
  ).join('');

  const tbody = document.getElementById('bench-tbody');
  tbody.innerHTML = BENCH.cases.map(c => {
    const cellsHtml = BENCH.columns.map(col => {
      const cell = c.cells[col.id];
      if (!cell || !cell.ready) {
        return `<td>
          <div class="cell-empty">
            <div style="margin-bottom:8px;">尚未生成</div>
            <button class="btn sm" onclick="runSingleCell('${c.id}','${col.id}', this)">生成此格</button>
          </div>
        </td>`;
      }
      const rtfCls = cell.rtf <= 1.0 ? 'rtf-fast' : 'rtf-mid';
      const chunkBadge = (cell.chunked && cell.chunk_count > 1)
        ? `<span class="metric rtf-fast">分句 <strong>${cell.chunk_count}段</strong> · 首句 <strong>${cell.ttfb_sec}s</strong></span>`
        : '';
      const chunkDetails = (cell.chunked && cell.chunks && cell.chunks.length > 1)
        ? `<details style="font-size:11px;color:var(--muted);margin-top:2px;"><summary style="cursor:pointer;">查看 ${cell.chunks.length} 段切分明细</summary><div style="margin-top:4px;display:flex;flex-direction:column;gap:3px;">${cell.chunks.map(ch => `<div>#${ch.index} (${ch.audio_sec}s / 算${ch.total_sec}s): ${escapeHtml(ch.text)}</div>`).join('')}</div></details>`
        : '';
      return `<td>
        <div class="cell-box">
          <audio controls preload="none" src="${escapeHtml(cell.audio_url)}"></audio>
          <div class="cell-meta">
            <span class="metric ${rtfCls}">RTF <strong>${cell.rtf}×</strong></span>
            <span class="metric">耗时 <strong>${cell.total_sec}s</strong></span>
            <span class="metric">时长 <strong>${cell.audio_sec}s</strong></span>
            <span class="metric">${escapeHtml(cell.device.toUpperCase())}${cell.threads ? ' ' + cell.threads + 'T' : ''}</span>
            ${chunkBadge}
          </div>
          ${renderStageBar(cell.stage_timings)}
          ${chunkDetails}
          <div style="display:flex;gap:6px;margin-top:2px;">
            <button class="btn sm" onclick="runSingleCell('${c.id}','${col.id}', this, true)">重跑</button>
            <a class="btn sm" style="text-decoration:none;" href="${escapeHtml(cell.audio_url)}" download="${c.id}_${col.id}.wav">WAV</a>
          </div>
        </div>
      </td>`;
    }).join('');
    return `<tr>
      <td class="prompt-col">
        <span class="case-tag">${escapeHtml(c.tag)}</span>
        <div class="case-title">${escapeHtml(c.title)}</div>
        <div class="case-text">${escapeHtml(c.text)}</div>
        <div class="case-cap">${escapeHtml(c.caption)}</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">
          <button class="btn sm" onclick="runBenchRow('${c.id}')">补齐本行</button>
          <button class="btn sm" onclick="loadPresetToStudio('${c.id}')">载入工作台</button>
        </div>
      </td>
      ${cellsHtml}
    </tr>`;
  }).join('');
}

async function runSingleCell(caseId, colId, btn, force = false) {
  showErr('');
  if (btn) { btn.disabled = true; btn.textContent = '推理中…'; }
  try {
    const res = await fetch('/api/bench/run_cell', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({case_id: caseId, col_id: colId, force})
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '推理失败');
    await refreshBench();
    fetchStatus();
  } catch (e) {
    showErr(e.message);
    if (btn) { btn.disabled = false; btn.textContent = '重试'; }
  }
}

async function runBenchRow(caseId) {
  showErr('');
  await fetch('/api/bench/run_batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({case_ids: [caseId]})
  });
  await refreshBench();
}

async function runBenchBatch(colIds) {
  showErr('');
  const res = await fetch('/api/bench/run_batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({col_ids: colIds})
  });
  const data = await res.json();
  if (!data.started && data.reason === 'nothing_to_run') {
    showErr('');
  }
  await refreshBench();
}

function toggleChunkOptions() {
  const checked = document.getElementById('auto-chunk').checked;
  document.getElementById('chunk-opts').style.opacity = checked ? '1' : '0.45';
}

function getFormPayload() {
  return {
    speaker_id: document.getElementById('speaker-select').value,
    text: document.getElementById('text-input').value,
    caption: document.getElementById('caption-input').value,
    model_id: document.getElementById('model-select').value,
    device: document.getElementById('device-select').value,
    num_steps: parseInt(document.getElementById('steps-select').value, 10),
    cpu_threads: parseInt(document.getElementById('threads-select').value, 10),
    seed: document.getElementById('seed-input').value.trim(),
    duration_scale: parseFloat(document.getElementById('dur-scale').value || '1.0'),
    auto_chunk: document.getElementById('auto-chunk').checked,
    max_chars: parseInt(document.getElementById('max-chars').value || '45', 10),
    pause_ms: parseInt(document.getElementById('pause-ms').value || '220', 10),
  };
}

async function runSynthesize() {
  showErr('');
  const btn = document.getElementById('btn-gen');
  const st = document.getElementById('studio-status');
  btn.disabled = true;
  st.textContent = '正在执行推理，请稍候…';
  try {
    const res = await fetch('/api/synthesize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(getFormPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '推理失败');
    st.textContent = `完成：时长 ${data.audio_sec}s · 耗时 ${data.total_sec}s · RTF ${data.rtf}× (${data.device.toUpperCase()})`;
    await loadHistory();
    fetchStatus();
  } catch (e) {
    showErr(e.message);
    st.textContent = '失败：' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function runCompare() {
  showErr('');
  const btn = document.getElementById('btn-cmp');
  const st = document.getElementById('studio-status');
  btn.disabled = true;
  st.textContent = '正在执行三路同台对比（clarity_fix 4步 CPU → 8步 CPU → Teacher RF-40），约需 10–15 秒…';
  try {
    const p = getFormPayload();
    const res = await fetch('/api/compare', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(p),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || '对比失败');
    st.textContent = `三路对比完成（seed=${data.seed}）`;
    const area = document.getElementById('compare-area');
    area.innerHTML = `<div class="result-item fresh">
      <div class="result-head">
        <div class="result-title">同台对比：${escapeHtml(data.speaker_ja)} · seed=${data.seed}</div>
      </div>
      <div class="result-text">${escapeHtml(data.text)}</div>
      <div class="result-cap">${escapeHtml(data.caption || '')}</div>
      <div class="compare-grid">
        ${data.items.map(it => `
          <div class="compare-card">
            <div style="font-weight:700;font-size:13px;margin-bottom:6px;">${escapeHtml(it.label)}</div>
            <audio controls src="${escapeHtml(it.audio_url)}"></audio>
            <div class="metrics" style="margin-top:6px;">
              <span class="metric ${it.rtf <= 1.0 ? 'rtf-fast' : 'rtf-mid'}">RTF <strong>${it.rtf}×</strong></span>
              <span class="metric">耗时 <strong>${it.total_sec}s</strong></span>
              <span class="metric">时长 <strong>${it.audio_sec}s</strong></span>
              ${it.chunked && it.chunk_count > 1 ? `<span class="metric rtf-fast">分句 <strong>${it.chunk_count}段</strong> · 首句 <strong>${it.ttfb_sec}s</strong></span>` : ''}
            </div>
            ${renderStageBar(it.stage_timings)}
            ${it.chunked && it.chunks && it.chunks.length > 1 ? `<details style="font-size:11px;color:var(--muted);margin-top:4px;"><summary style="cursor:pointer;">切分明细 (${it.chunks.length}段)</summary><div style="margin-top:4px;display:flex;flex-direction:column;gap:2px;">${it.chunks.map(ch => `<div>#${ch.index} (${ch.audio_sec}s / 算${ch.total_sec}s): ${escapeHtml(ch.text)}</div>`).join('')}</div></details>` : ''}
          </div>
        `).join('')}
      </div>
    </div>`;
    await loadHistory();
    fetchStatus();
  } catch (e) {
    showErr(e.message);
    st.textContent = '对比失败：' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function loadHistory() {
  const res = await fetch('/api/history');
  const items = await res.json();
  document.getElementById('hist-count').textContent = `${items.length} 条记录`;
  const list = document.getElementById('history-list');
  if (!items.length) {
    list.innerHTML = '<div class="muted">暂无实时合成记录。</div>';
    return;
  }
  list.innerHTML = items.map((it, idx) => `
    <div class="result-item ${idx === 0 ? 'fresh' : ''}">
      <div class="result-head">
        <div class="result-title">${escapeHtml(it.speaker_ja || it.speaker_id)} · ${escapeHtml(it.model_id)} (${it.num_steps} 步${it.chunked && it.chunk_count > 1 ? ` · 分句${it.chunk_count}段` : ''})</div>
        <div class="metrics">
          <span class="metric ${it.rtf <= 1.0 ? 'rtf-fast' : 'rtf-mid'}">RTF <strong>${it.rtf}×</strong></span>
          <span class="metric">耗时 <strong>${it.total_sec}s</strong></span>
          <span class="metric">时长 <strong>${it.audio_sec}s</strong></span>
          ${it.chunked && it.chunk_count > 1 ? `<span class="metric rtf-fast">首句 <strong>${it.ttfb_sec}s</strong></span>` : ''}
          <span class="metric">${escapeHtml(it.device.toUpperCase())}${it.threads ? ' ' + it.threads + 'T' : ''}</span>
          <span class="metric">seed ${it.used_seed}</span>
        </div>
      </div>
      <div class="result-text">${escapeHtml(it.text)}</div>
      ${it.caption ? `<div class="result-cap">${escapeHtml(it.caption)}</div>` : ''}
      <audio controls preload="metadata" src="${escapeHtml(it.audio_url)}"></audio>
      ${renderStageBar(it.stage_timings)}
      ${it.chunked && it.chunks && it.chunks.length > 1 ? `<details style="font-size:11.5px;color:var(--muted);margin-top:6px;"><summary style="cursor:pointer;">查看 ${it.chunks.length} 段切分与逐句耗时</summary><div style="margin-top:4px;display:flex;flex-direction:column;gap:3px;">${it.chunks.map(ch => `<div>#${ch.index} (时长 ${ch.audio_sec}s · CPU ${ch.total_sec}s · RTF ${ch.rtf}×): ${escapeHtml(ch.text)}</div>`).join('')}</div></details>` : ''}
      <div class="muted" style="margin-top:6px;display:flex;justify-content:space-between;">
        <span>${escapeHtml(it.created_at)}</span>
        <a style="color:var(--accent);text-decoration:none;" href="${escapeHtml(it.audio_url)}" download="${escapeHtml(it.filename)}">下载 48kHz WAV</a>
      </div>
    </div>
  `).join('');
}

(async function init() {
  await fetchStatus();
  await Promise.all([refreshBench(), loadHistory()]);
})();
</script>
</body>
</html>
"""


def make_handler(engine: EngineManager):
    class WorkbenchHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            # Keep logs concise
            sys.stderr.write(f"[http] {self.address_string()} - {fmt % args}\n")

        def _send_json(self, obj: Any, status: int = 200) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _send_html(self, body: str) -> None:
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _send_audio_file(self, path: Path) -> None:
            if not path.is_file():
                self._send_json({"error": "File not found"}, status=404)
                return
            size = path.stat().st_size
            range_hdr = self.headers.get("Range")
            start = 0
            end = size - 1
            status = 200
            if range_hdr and range_hdr.startswith("bytes="):
                spec = range_hdr[6:].split(",")[0].strip()
                parts = spec.split("-")
                if parts[0]:
                    start = int(parts[0])
                if len(parts) > 1 and parts[1]:
                    end = min(size - 1, int(parts[1]))
                if start >= size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                status = 206
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "public, max-age=3600")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path in ("/", "/index.html"):
                self._send_html(INDEX_HTML)
                return
            if path == "/api/status":
                self._send_json(engine.status_dict())
                return
            if path == "/api/bench/status":
                self._send_json(engine.get_bench_state())
                return
            if path == "/api/history":
                self._send_json(load_history())
                return
            if path.startswith("/api/history_audio/"):
                fname = Path(path.removeprefix("/api/history_audio/")).name
                self._send_audio_file(HISTORY_DIR / fname)
                return
            if path.startswith("/api/bench_audio/"):
                rel = path.removeprefix("/api/bench_audio/").strip("/")
                parts = rel.split("/")
                if len(parts) == 2:
                    case_id = Path(parts[0]).name
                    fname = Path(parts[1]).name
                    self._send_audio_file(BENCH_DIR / case_id / fname)
                    return
                self._send_json({"error": "Invalid path"}, status=400)
                return
            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            try:
                body = self._read_json_body()
            except Exception as exc:
                self._send_json({"error": f"Invalid JSON: {exc}"}, status=400)
                return

            try:
                if path == "/api/synthesize":
                    speaker_id = str(body.get("speaker_id", "fuyuko"))
                    text = str(body.get("text", ""))
                    caption = str(body.get("caption", ""))
                    model_id = str(body.get("model_id", "clarity_fix"))
                    device = str(body.get("device", "cpu"))
                    num_steps = int(body.get("num_steps", 4))
                    cpu_threads = body.get("cpu_threads")
                    seed_raw = body.get("seed", 42)
                    seed = None if (seed_raw is None or str(seed_raw).strip() == "") else int(seed_raw)
                    duration_scale = float(body.get("duration_scale", 1.0))
                    auto_chunk = bool(body.get("auto_chunk", False))
                    max_chars = int(body.get("max_chars", 45))
                    pause_ms = int(body.get("pause_ms", 220))

                    res = engine.synthesize(
                        speaker_id=speaker_id,
                        text=text,
                        caption=caption,
                        model_id=model_id,
                        device=device,
                        num_steps=num_steps,
                        seed=seed,
                        duration_scale=duration_scale,
                        cpu_threads=int(cpu_threads) if cpu_threads else None,
                        auto_chunk=auto_chunk,
                        max_chars=max_chars,
                        pause_ms=pause_ms,
                    )
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    uid = uuid.uuid4().hex[:6]
                    chunk_tag = f"_c{res['chunk_count']}" if res.get("chunked") else ""
                    fname = f"{ts}_{speaker_id}_{model_id}_s{res['num_steps']}{chunk_tag}_{uid}.wav"
                    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
                    save_wav(HISTORY_DIR / fname, res["audio"], res["sample_rate"])
                    sp = SPEAKER_BY_ID.get(speaker_id, {"ja": speaker_id})
                    entry = {
                        "id": uid,
                        "filename": fname,
                        "audio_url": f"/api/history_audio/{fname}",
                        "speaker_id": speaker_id,
                        "speaker_ja": sp["ja"],
                        "text": text,
                        "caption": caption,
                        "model_id": model_id,
                        "device": res["device"],
                        "num_steps": res["num_steps"],
                        "threads": res["threads"],
                        "audio_sec": res["audio_sec"],
                        "total_sec": res["total_sec"],
                        "rtf": res["rtf"],
                        "used_seed": res["used_seed"],
                        "stage_timings": res["stage_timings"],
                        "chunked": res.get("chunked", False),
                        "chunk_count": res.get("chunk_count", 1),
                        "ttfb_sec": res.get("ttfb_sec", res["total_sec"]),
                        "chunks": res.get("chunks", []),
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    append_history(entry)
                    self._send_json(entry)
                    return

                if path == "/api/compare":
                    speaker_id = str(body.get("speaker_id", "fuyuko"))
                    text = str(body.get("text", ""))
                    caption = str(body.get("caption", ""))
                    cpu_threads = int(body.get("cpu_threads", engine._default_threads))
                    seed_raw = body.get("seed", 42)
                    seed = int(secrets.randbelow(2**31 - 1)) if (seed_raw is None or str(seed_raw).strip() == "") else int(seed_raw)
                    duration_scale = float(body.get("duration_scale", 1.0))
                    auto_chunk = bool(body.get("auto_chunk", True))
                    max_chars = int(body.get("max_chars", 45))
                    pause_ms = int(body.get("pause_ms", 220))

                    will_split = auto_chunk and len(split_japanese_chunks(text, max_chars=max_chars)) > 1
                    if will_split:
                        specs = [
                            ("clarity_fix 4步 整段 (CPU)", "clarity_fix", "cpu", 4, False),
                            ("clarity_fix 4步 分句拼接 (CPU)", "clarity_fix", "cpu", 4, True),
                            ("clarity_fix 8步 分句拼接 (CPU)", "clarity_fix", "cpu", 8, True),
                            ("Teacher s8000 RF-40", "teacher_s8000", "auto_gpu", 40, False),
                        ]
                    else:
                        specs = [
                            ("clarity_fix 4步 (CPU)", "clarity_fix", "cpu", 4, False),
                            ("clarity_fix 8步 (CPU)", "clarity_fix", "cpu", 8, False),
                            ("Teacher s8000 RF-40", "teacher_s8000", "auto_gpu", 40, False),
                        ]
                    sp = SPEAKER_BY_ID.get(speaker_id, {"ja": speaker_id})
                    items_out: list[dict[str, Any]] = []
                    for label, mid, dev, steps, use_chunk in specs:
                        res = engine.synthesize(
                            speaker_id=speaker_id,
                            text=text,
                            caption=caption,
                            model_id=mid,
                            device=dev,
                            num_steps=steps,
                            seed=seed,
                            duration_scale=duration_scale,
                            cpu_threads=cpu_threads,
                            auto_chunk=use_chunk,
                            max_chars=max_chars,
                            pause_ms=pause_ms,
                        )
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        uid = uuid.uuid4().hex[:6]
                        chunk_tag = f"_c{res['chunk_count']}" if res.get("chunked") else ""
                        fname = f"{ts}_{speaker_id}_{mid}_s{res['num_steps']}{chunk_tag}_{uid}.wav"
                        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
                        save_wav(HISTORY_DIR / fname, res["audio"], res["sample_rate"])
                        entry = {
                            "id": uid,
                            "label": label,
                            "filename": fname,
                            "audio_url": f"/api/history_audio/{fname}",
                            "speaker_id": speaker_id,
                            "speaker_ja": sp["ja"],
                            "text": text,
                            "caption": caption,
                            "model_id": mid,
                            "device": res["device"],
                            "num_steps": res["num_steps"],
                            "threads": res["threads"],
                            "audio_sec": res["audio_sec"],
                            "total_sec": res["total_sec"],
                            "rtf": res["rtf"],
                            "used_seed": res["used_seed"],
                            "stage_timings": res["stage_timings"],
                            "chunked": res.get("chunked", False),
                            "chunk_count": res.get("chunk_count", 1),
                            "ttfb_sec": res.get("ttfb_sec", res["total_sec"]),
                            "chunks": res.get("chunks", []),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        append_history(entry)
                        items_out.append(entry)
                    self._send_json({
                        "speaker_id": speaker_id,
                        "speaker_ja": sp["ja"],
                        "text": text,
                        "caption": caption,
                        "seed": seed,
                        "items": items_out,
                    })
                    return

                if path == "/api/bench/run_cell":
                    case_id = str(body["case_id"])
                    col_id = str(body["col_id"])
                    force = bool(body.get("force", False))
                    meta = engine.run_bench_cell(case_id, col_id, force=force)
                    self._send_json(meta)
                    return

                if path == "/api/bench/run_batch":
                    case_ids = body.get("case_ids")
                    col_ids = body.get("col_ids")
                    force = bool(body.get("force", False))
                    resp = engine.start_bench_batch(
                        case_ids=case_ids,
                        col_ids=col_ids,
                        force=force,
                    )
                    self._send_json(resp)
                    return

                self._send_json({"error": "Not found"}, status=404)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

    return WorkbenchHandler


def main() -> int:
    parser = argparse.ArgumentParser(description="SC29 MeanFlow CPU Inference & Acceptance Workbench")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--threads", type=int, default=10)
    parser.add_argument(
        "--pregen-cols",
        default="",
        help="Optional comma-separated benchmark column IDs to pre-generate in background on startup",
    )
    parser.add_argument(
        "--pregen-cases",
        default="",
        help="Optional comma-separated benchmark case IDs to pre-generate in background on startup",
    )
    args = parser.parse_args()

    WORKBENCH_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    engine = EngineManager(default_threads=args.threads)

    def _startup_worker() -> None:
        engine.preload_default_cpu()
        if args.pregen_cols.strip():
            cols = [c.strip() for c in args.pregen_cols.split(",") if c.strip()]
            cases = [c.strip() for c in args.pregen_cases.split(",") if c.strip()] or None
            engine.start_bench_batch(case_ids=cases, col_ids=cols, force=False)

    threading.Thread(target=_startup_worker, daemon=True).start()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    print(f"[workbench] listening on http://{args.host}:{args.port} (default CPU threads={args.threads})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
