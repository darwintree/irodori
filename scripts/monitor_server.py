#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

LOG_FILE = Path(
    "/home/darwinwsl/.gemini/antigravity-cli/brain/a5bb3a90-52f7-4bab-990b-6ccbe32ba6b7/.system_generated/tasks/task-826.log"
)
OUTPUT_DIR = Path("/home/darwinwsl/code/irodori/outputs/mano_lora_captioned")
NO_CAPTION_LORA = Path("/home/darwinwsl/code/irodori/outputs/mano_lora_baseline/checkpoint_0003000")
FULL_NOCAP_CKPT = Path("/home/darwinwsl/code/irodori/outputs/mano_full_nocap/checkpoint_best_val_loss_0001000_0.728665.pt")
FULL_CAP_CKPT = Path("/home/darwinwsl/code/irodori/outputs/mano_full_captioned/checkpoint_best_val_loss_0001000_0.710021.pt")
EVAL_DIR = OUTPUT_DIR / "eval_samples"
BASE_CKPT = Path("/home/darwinwsl/code/irodori/models/Irodori-TTS-v4.1-Small/model.safetensors")
IRODORI_REPO = Path("/home/darwinwsl/code/irodori/Irodori-TTS")
PYTHON_BIN = IRODORI_REPO / ".venv/bin/python"

REF_WAVS = [
    Path("/home/darwinwsl/code/irodori/datasets/mano/wavs/mano_00000.wav"),
    Path("/home/darwinwsl/code/irodori/datasets/mano/wavs/mano_00002.wav"),
    Path("/home/darwinwsl/code/irodori/datasets/mano/wavs/mano_00003.wav"),
]
SHORT_REF_WAV = Path("/home/darwinwsl/code/irodori/datasets/mano/wavs/mano_00000.wav")
GT_SAMPLES = [
    (
        "mano_05542.wav",
        "（……今日のレッスン、ダメダメだったな……）",
        "落ち込んで自信をなくしたように、静かで力のない声で呟いている。",
    ),
    (
        "mano_05543.wav",
        "……！　うん、頑張ろうね、灯織ちゃん！",
        "ハッとしたあと、明るく前向きに元気づけるように弾んだ口調で話している。",
    ),
]

AMBIG_A = "……本当に、それでいいんですか？"
AMBIG_B = "プロデューサーさん、ありがとうございます。"

# (filename, category, text, caption, ref_mode: "full" | "short" | "none")
TEST_PROMPTS = [
    # 多向解读 A（5种强反差表演）
    ("01_ambigA_gentle.wav", "多向解读 A (5向)", AMBIG_A, "落ち着いた柔らかな口調で、相手を気遣いながら静かに尋ねている。", "full"),
    ("02_ambigA_teary.wav", "多向解读 A (5向)", AMBIG_A, "涙を堪えながら声を小さく震わせ、今にも消えそうな音量でおずおずと尋ねている。", "full"),
    ("03_ambigA_angry.wav", "多向解读 A (5向)", AMBIG_A, "怒りを抑えた冷たく強い口調で、語尾を言い切るようにきっぱりと問い詰めている。", "full"),
    ("04_ambigA_surp.wav", "多向解读 A (5向)", AMBIG_A, "息を呑んで驚いたように、やや早口で勢いよく聞き返している。", "full"),
    ("05_ambigA_dark_act.wav", "多向解读 A (5向)", AMBIG_A, "低く落ち着いた妖艶なトーンで、ゆっくりと間を取りながら試すように問いかけている。", "full"),
    # 多向解读 B（4种强反差表演）
    ("06_ambigB_happy.wav", "多向解读 B (4向)", AMBIG_B, "明るく弾んだ口調で、心から嬉しそうに元気よくお礼を伝えている。", "full"),
    ("07_ambigB_crying.wav", "多向解读 B (4向)", AMBIG_B, "涙に詰まって言葉を途切れさせながら、深い感謝を込めて震える声で静かに伝えている。", "full"),
    ("08_ambigB_whisper.wav", "多向解读 B (4向)", AMBIG_B, "少し照れくさそうに息を漏らしながら、控えめな囁き声で優しくお礼を言っている。", "full"),
    ("09_ambigB_exhausted.wav", "多向解读 B (4向)", AMBIG_B, "疲れ切ったように力を抜いた平坦なトーンで、無理に明るく振る舞うように淡々と伝えている。", "full"),
    # 情绪不匹配对抗（4组）
    ("10_mismatch_happy2cry.wav", "情绪不匹配对抗", "わあ、すごい！とっても素敵ですね！", "涙を堪えながら力のない震えた声で、無理に喜んでいるように苦しげに話している。", "full"),
    ("11_mismatch_sad2playful.wav", "情绪不匹配对抗", "ごめんなさい……私のせいで、全部台無しに……", "あえて軽く明るい調子で、少し早口におどけながらごまかすように話している。", "full"),
    ("12_mismatch_calm2panic.wav", "情绪不匹配对抗", "今日のレッスンは午後三時からですね。", "息を切らして慌てたように、早口で焦りを滲ませながら必死に伝えている。", "full"),
    ("13_mismatch_soft2shout.wav", "情绪不匹配对抗", "私、そんなこと望んでいません……！", "拒絶するように声を荒らげて激しく張り上げ、必死に強く否定している。", "full"),
    # 句内动态情绪转折（2组）
    ("14_shift_timid2firm.wav", "句内情绪转折", "……私なんかがセンターでいいのかなって、ずっと不安でした。でも、今はみんなと一緒に輝きたいんですっ！", "前半は自信なさげに控えめな小声で呟き、後半は一転して力を込めてまっすぐ力強く宣言している。", "full"),
    ("15_shift_laugh2warm.wav", "句内情绪转折", "ふふ……なんだか、すごく懐かしいですね。あの公園で出会った日のこと、今でもはっきり覚えています。", "小さく柔らかな笑い声を漏らしたあと、懐かしむようにややゆっくりと温かく語りかけている。", "full"),
    # 长句与极端发声（2组）
    ("16_long_stage.wav", "长句与极端发声", "今日のステージは、今までみんなで練習してきた集大成だから、最後まで笑顔で精一杯届けたいと思います。イルミネーションスターズの輝きを、見ていてくださいね。", "一語一語をはっきりと丁寧に区切りながら、凛とした張りのある口調で想いを込めて語りかけている。", "full"),
    ("17_extreme_whisper.wav", "长句与极端发声", "……しっ、静かにしてください。ほら、あそこに小鳥さんが休んでいますよ……", "相手を気遣うように極小の囁き声で、息を多めに含ませながらそっと話している。", "full"),
    # 参考音频压力测试 & 无参考仅 Caption 对比（9组）
    ("18_shortref_happy.wav", "参考压力测试", AMBIG_A, "明るく弾んだ口調で、楽しそうに尋ねている。", "short"),
    ("19_noref_happy.wav", "无参考仅 Caption", AMBIG_A, "明るく弾んだ口調で、楽しそうに尋ねている。", "none"),
    ("20_noref_teary.wav", "无参考仅 Caption", AMBIG_A, "涙を堪えながら声を小さく震わせ、おずおずと控えめに尋ねている。", "none"),
    ("23_noref_angry.wav", "无参考仅 Caption", AMBIG_A, "怒りを抑えた冷たく強い口調で、語尾を言い切るようにきっぱりと問い詰めている。", "none"),
    ("24_noref_whisper.wav", "无参考仅 Caption", "……しっ、静かにしてください。ほら、あそこに小鳥さんが休んでいますよ……", "相手を気遣うように極小の囁き声で、息を多めに含ませながらそっと話している。", "none"),
    ("25_noref_crying.wav", "无参考仅 Caption", AMBIG_B, "涙に詰まって言葉を途切れさせながら、深い感謝を込めて震える声で静かに伝えている。", "none"),
    ("26_noref_shout.wav", "无参考仅 Caption", "私、そんなこと望んでいません……！", "拒絶するように声を荒らげて激しく張り上げ、必死に強く否定している。", "none"),
    ("27_noref_shift.wav", "无参考仅 Caption", "……私なんかがセンターでいいのかなって、ずっと不安でした。でも、今はみんなと一緒に輝きたいんですっ！", "前半は自信なさげに控えめな小声で呟き、後半は一転して力を込めてまっすぐ力強く宣言している。", "none"),
    ("28_noref_dark_act.wav", "无参考仅 Caption", "血を頂いても、いいですか……？", "低く落ち着いた妖艶なトーンで、ゆっくりと間を取りながら試すように問いかけている。", "none"),
    # 新增：无参考 OOD 极简关键词 & 极端反差 & 无参考情绪对抗（10组）
    ("29_noref_kw_cry.wav", "无参考 OOD 短词", AMBIG_B, "泣きながら、震える声", "none"),
    ("30_noref_kw_angry.wav", "无参考 OOD 短词", AMBIG_A, "すごく怒ってる、強い口調", "none"),
    ("31_noref_kw_sleepy.wav", "无参考 OOD 短词", "プロデューサーさん……もう、こんな時間なんですね……", "眠そうに、ぼんやりと、ゆっくり", "none"),
    ("32_noref_kw_fast_panic.wav", "无参考 OOD 短词", "大変ですっ！レッスン室の鍵が見当たらなくて……！", "焦って早口、パニック", "none"),
    ("33_noref_tsundere.wav", "无参考极端反差", "べ、別にプロデューサーさんのために作ったわけじゃないんですから……！", "少し照れ隠しながらもツンとした強気な口調で、早口に言い訳している。", "none"),
    ("34_noref_scorn.wav", "无参考极端反差", "へぇ……それ、本気で言ってるんですか？", "呆れて馬鹿にしたように、冷たく鼻で笑いながら見下す口調で話している。", "none"),
    ("35_noref_terror.wav", "无参考极端反差", "だ、誰か……そこにいるんですか……？来ないで……！", "恐怖に怯えて息を震わせながら、今にも泣き出しそうな切迫した声で叫んでいる。", "none"),
    ("36_noref_cheer_loud.wav", "无参考极端反差", "みんなーっ！今日は最後まで一緒に盛り上がっていこうねーっ！！", "広い会場に向かって声を張り上げ、全力で明るく元気いっぱいに呼びかけている。", "none"),
    ("37_noref_mismatch_happy2dead.wav", "无参考情绪对抗", "わあ、すごい！とっても素敵ですね！", "感情を完全に失ったように棒読みで、低く冷たく淡々と話す。", "none"),
    ("38_noref_mismatch_sad2laugh.wav", "无参考情绪对抗", "もうダメです……全部、終わっちゃいました……", "こらえきれず吹き出しそうに笑いながら、楽しげで陽気な調子で話している。", "none"),
    # GT 原声对照（2组）
    ("21_gt_05542.wav", "GT 原声对照", GT_SAMPLES[0][1], GT_SAMPLES[0][2], "full"),
    ("22_gt_05543.wav", "GT 原声对照", GT_SAMPLES[1][1], GT_SAMPLES[1][2], "full"),
]

TRAIN_RE = re.compile(
    r"step=(\d+)\s+loss=([0-9.]+)\s+rf=([0-9.]+)\s+dur=([0-9.]+)\s+dur_mae=([0-9.]+)\s+dur_bias=([-0-9.]+)\s+lr=([0-9.eE+-]+)"
)
VALID_RE = re.compile(
    r"valid\s+(?:final\s+)?step=(\d+)\s+loss=([0-9.]+)\s+rf=([0-9.]+)\s+dur=([0-9.]+)\s+dur_mae=([0-9.]+)\s+dur_bias=([-0-9.]+)"
)
PROGRESS_RE = re.compile(
    r"Train RF:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[([^\r\n\]]+)"
)

_eval_status_lock = threading.Lock()
_currently_evaluating: str | None = None


def parse_log() -> dict:
    train_pts = []
    valid_pts = []
    progress = None
    if LOG_FILE.exists():
        text = LOG_FILE.read_text(encoding="utf-8", errors="ignore")
        for m in TRAIN_RE.finditer(text):
            train_pts.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "rf": float(m.group(3)),
                "dur": float(m.group(4)),
                "dur_mae": float(m.group(5)),
                "dur_bias": float(m.group(6)),
                "lr": float(m.group(7)),
            })
        for m in VALID_RE.finditer(text):
            valid_pts.append({
                "step": int(m.group(1)),
                "loss": float(m.group(2)),
                "rf": float(m.group(3)),
                "dur": float(m.group(4)),
                "dur_mae": float(m.group(5)),
                "dur_bias": float(m.group(6)),
            })
        # find last progress bar match
        matches = list(PROGRESS_RE.finditer(text))
        if matches:
            last = matches[-1]
            progress = {
                "pct": int(last.group(1)),
                "step": int(last.group(2)),
                "total": int(last.group(3)),
                "timing": last.group(4),
            }
    return {
        "train": train_pts,
        "valid": valid_pts,
        "progress": progress,
    }


def get_gpu_info() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) == 5:
            return f"VRAM {parts[0]}/{parts[1]} MiB | Util {parts[2]}% | Power {parts[3]}W | Temp {parts[4]}°C"
    except Exception:
        pass
    return "GPU info unavailable"


def list_eval_groups() -> list[dict]:
    groups = []
    if not EVAL_DIR.exists():
        return groups
    for d in sorted(EVAL_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "done.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        groups.append(meta)
    # Put base_zero_shot at the end or sorted by step descending
    groups.sort(key=lambda g: g.get("step", -1), reverse=True)
    return groups


def run_eval_for(name: str, step: int, lora_dir: Path | None, ckpt_path: Path | None = None) -> None:
    global _currently_evaluating
    target_dir = EVAL_DIR / name
    done_file = target_dir / "done.json"
    if done_file.exists():
        try:
            existing = json.loads(done_file.read_text(encoding="utf-8"))
            if len(existing.get("items", [])) == len(TEST_PROMPTS):
                return
        except Exception:
            pass
    target_dir.mkdir(parents=True, exist_ok=True)
    with _eval_status_lock:
        _currently_evaluating = name

    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    items = []
    jobs = []
    try:
        for fname, category, prompt_text, prompt_caption, ref_mode in TEST_PROMPTS:
            out_wav = target_dir / fname
            jobs.append({
                "out_wav": str(out_wav),
                "text": prompt_text,
                "caption": prompt_caption,
                "ref_mode": ref_mode,
                "ref_wav": str(SHORT_REF_WAV),
                "ref_wavs": [str(p) for p in REF_WAVS],
            })
            items.append({
                "file": f"{name}/{fname}",
                "category": category,
                "text": prompt_text,
                "caption": prompt_caption,
                "ref_mode": ref_mode,
            })

        if any(not Path(j["out_wav"]).exists() for j in jobs):
            jobs_file = target_dir / "_jobs.json"
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            cmd = [
                str(PYTHON_BIN),
                "/home/darwinwsl/code/irodori/scripts/batch_infer_eval.py",
                "--checkpoint", str(ckpt_path or BASE_CKPT),
                "--jobs-json", str(jobs_file),
            ]
            if lora_dir is not None:
                cmd.extend(["--lora-adapter", str(lora_dir)])
            subprocess.run(cmd, cwd=str(IRODORI_REPO), env=env, check=True)
            jobs_file.unlink(missing_ok=True)
        done_file.write_text(
            json.dumps({"name": name, "step": step, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[eval] failed for {name}: {exc}", flush=True)
    finally:
        with _eval_status_lock:
            _currently_evaluating = None


def evaluator_loop() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ref_out = EVAL_DIR / "refs"
    ref_out.mkdir(parents=True, exist_ok=True)
    for p in [*REF_WAVS, *[Path("/home/darwinwsl/code/irodori/datasets/mano/wavs") / g[0] for g in GT_SAMPLES]]:
        dst = ref_out / p.name
        if not dst.exists() and p.exists():
            shutil.copy2(p, dst)

    # First generate base model zero-shot baseline, previous no-caption LoRA, and Full FT A/B models
    run_eval_for("base_zero_shot", 0, None)
    if NO_CAPTION_LORA.exists():
        run_eval_for("no_caption_lora_3000", 1, NO_CAPTION_LORA)
    if FULL_NOCAP_CKPT.exists():
        run_eval_for("full_nocap_1000", 2000, None, ckpt_path=FULL_NOCAP_CKPT)
    if FULL_CAP_CKPT.exists():
        run_eval_for("full_captioned_1000", 2001, None, ckpt_path=FULL_CAP_CKPT)

    while True:
        try:
            if OUTPUT_DIR.exists():
                ckpts = sorted(
                    [
                        p for p in OUTPUT_DIR.iterdir()
                        if p.is_dir() and p.name.startswith("checkpoint_") and (p / "adapter_model.safetensors").exists()
                    ],
                    reverse=True,
                )
                for ckpt in ckpts:
                    m = re.match(r"checkpoint_(?:best_val_loss_)?(\d+)", ckpt.name)
                    if m:
                        step = int(m.group(1))
                        eval_name = f"captioned_step_{step:07d}"
                    elif ckpt.name == "checkpoint_final":
                        step = 1000
                        eval_name = "captioned_step_0001000"
                    else:
                        continue
                    if not (EVAL_DIR / eval_name / "done.json").exists():
                        time.sleep(2)
                        run_eval_for(eval_name, step, ckpt)
                    else:
                        try:
                            ex = json.loads((EVAL_DIR / eval_name / "done.json").read_text(encoding="utf-8"))
                            if len(ex.get("items", [])) != len(TEST_PROMPTS):
                                run_eval_for(eval_name, step, ckpt)
                        except Exception:
                            pass
        except Exception as exc:
            print(f"[watcher] error: {exc}", flush=True)
        time.sleep(15)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Irodori-TTS 真乃 LoRA 跨模型对比台</title>
<style>
  :root {
    --bg: #f8fafc; --card: #fff; --border: #e2e8f0; --text: #0f172a; --muted: #475569;
    --accent: #2563eb; --soft: #f1f5f9;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 16px 20px 48px; background: var(--bg); color: var(--text); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 960px) { .row { grid-template-columns: 1fr; } }
  h2, h3 { margin: 0 0 8px 0; }
  .meta { font-size: 13px; color: var(--muted); }
  .bar-wrap { background: #e2e8f0; border-radius: 999px; height: 10px; overflow: hidden; margin: 6px 0; }
  .bar-fill { background: #3b82f6; height: 100%; transition: width .3s; }
  svg { width: 100%; height: 200px; background: #fcfcfd; border: 1px solid #f1f5f9; border-radius: 6px; }
  .controls { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0 12px; }
  .pill { background: var(--soft); border: 1px solid var(--border); color: var(--muted); padding: 4px 10px; border-radius: 999px; cursor: pointer; font-size: 12px; }
  .pill.active { background: #dbeafe; border-color: var(--accent); color: #1d4ed8; font-weight: 600; }
  .model-toggles { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 8px 12px; background: var(--soft); border-radius: 8px; font-size: 13px; }
  .model-toggles label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; font-weight: 600; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; table-layout: fixed; }
  th, td { border: 1px solid var(--border); padding: 10px; font-size: 13px; vertical-align: top; text-align: left; }
  th { background: var(--soft); position: sticky; top: 0; z-index: 2; }
  col.prompt-col { width: 340px; }
  col.model-col { width: 235px; }
  .prompt-text { font-size: 14px; font-weight: 600; margin: 5px 0; line-height: 1.45; }
  .prompt-cap { font-size: 12.5px; color: #1e40af; background: #eff6ff; border-left: 3px solid #3b82f6; padding: 5px 8px; border-radius: 4px; line-height: 1.4; }
  .prompt-cap-zh { font-size: 12px; color: #475569; margin-top: 3px; line-height: 1.35; }
  .gt-box { margin-top: 8px; padding-top: 6px; border-top: 1px dashed var(--border); }
  audio { height: 34px; width: 100%; min-width: 190px; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; background: #dbeafe; color: #1d4ed8; margin-right: 4px; }
  .badge.short { background: #e0e7ff; color: #3730a3; }
  .badge.none { background: #fef3c7; color: #92400e; }
  .badge.m-base { background: #e2e8f0; color: #334155; }
  .badge.m-nocap { background: #fee2e2; color: #991b1b; }
  .badge.m-cap { background: #dcfce7; color: #166534; }
  tr:hover td { background: #fafcff; }
  .verdict-card { border-left: 4px solid #16a34a; }
  .verdict-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; margin-top: 12px; }
  .cite-box { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .cite-title { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
  .cite-desc { font-size: 12.5px; color: var(--muted); line-height: 1.45; margin-bottom: 8px; }
  .cite-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .cite-pair audio { width: 100%; min-width: 140px; height: 32px; }
  .verdict-summary { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 14px; font-size: 13.5px; line-height: 1.6; margin-bottom: 12px; }
  details summary { cursor: pointer; font-weight: 600; font-size: 14px; color: var(--text); }
</style>
</head>
<body>

<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
    <h2>Irodori-TTS 真乃 LoRA 跨模型 Caption 对比台 <span id="evalBadge" class="badge" style="display:none"></span></h2>
    <div class="meta" id="gpuText">GPU: 加载中...</div>
  </div>
  <div class="meta" id="progressText">加载中...</div>
  <div class="bar-wrap"><div class="bar-fill" id="progressBar" style="width:0%"></div></div>
</div>

<div class="card">
  <h3>跨模型同句横向对比（点击任一播放器会自动暂停其他音频）</h3>
  <div class="model-toggles" id="modelToggles">加载模型列表中...</div>
  <div class="controls" id="catPills"></div>
  <div class="table-wrap" id="samplesArea"><div class="meta">等待样本加载...</div></div>
</div>

<div class="card">
  <details>
    <summary>训练曲线（Total / RF / Duration Loss）与参考原声</summary>
    <div class="row" style="margin-top:12px;">
      <div>
        <div class="meta" style="margin-bottom:4px;"><b>Total Loss &amp; RF Loss (Train / Valid)</b></div>
        <svg id="chartLoss" viewBox="0 0 560 200"></svg>
      </div>
      <div>
        <div class="meta" style="margin-bottom:4px;"><b>Duration Loss (Train / Valid)</b></div>
        <svg id="chartDur" viewBox="0 0 560 200"></svg>
      </div>
    </div>
    <div style="display:flex; gap:16px; flex-wrap:wrap; margin-top:12px;">
      <div><div class="meta">参考1: mano_00000.wav</div><audio controls preload="none" src="/samples/refs/mano_00000.wav" style="width:240px;"></audio></div>
      <div><div class="meta">参考2: mano_00002.wav</div><audio controls preload="none" src="/samples/refs/mano_00002.wav" style="width:240px;"></audio></div>
      <div><div class="meta">参考3: mano_00003.wav</div><audio controls preload="none" src="/samples/refs/mano_00003.wav" style="width:240px;"></audio></div>
    </div>
  </details>
</div>

<div class="card verdict-card">
  <h3>AI 听觉实测结论：全量微调「无 Caption」vs「有 Caption (12.6% 分层×3)」</h3>
  <div class="verdict-summary">
    <b>总体判断：存在明确且可稳定听出的实质差异，核心差距在「极端情绪响应」与「防止 Caption 控制力钝化」。</b><br>
    1. <b>音色记忆</b>：两版全量模型在 <code>--no-ref</code> 下均已将真乃音色稳固写入权重（对比 <code>base_zero_shot</code> 的泛化女声变化极大）。<br>
    2. <b>控制力差异</b>：<code>full_nocap_1000</code> 明显出现向真乃默认温柔声线坍缩的现象，对哭腔、冷峻质问与「文本-情绪对抗」指令响应偏弱甚至忽略；<code>full_captioned_1000</code> 仅靠 12.6% 分层标注（3× 过采样至 30.2%）就稳住了强情绪控制力，哽咽微颤、情绪对抗和句内转折均明显更贴合指令。<br>
    3. <b>差距较小的场景</b>：当台词本身已带有强语境提示词（如耳语「しっ」、戏中戏「血を頂いても」、强拒绝「望んでいません！」）或普通明朗语气时，两版全量都能较好完成，有标版仅在气声与节奏细节上略胜一筹。<br>
    4. <b>工程建议</b>：后续多角色全量微调<b>不需要 100% 标全库</b>，保持 <b>10%～15% 分层高质量标注（70% 高情绪 + 30% 日常）并在训练时过采样至 ~30%</b> 即是性价比最高的最优解。
  </div>

  <div class="verdict-grid">
    <div class="cite-box">
      <div class="cite-title">证据 1（差距最大）：#07 哽咽落泪道谢（有参考）</div>
      <div class="cite-desc">台词 <code>プロデューサーさん、ありがとうございます。</code>（中性礼貌词）。无标版几乎无视哭腔指令，仍偏明亮礼貌；有标版听出明显的压低哽咽与声音微颤。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/07_ambigB_crying.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/07_ambigB_crying.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">证据 2（情绪对抗）：#10 开心台词 + 痛苦哭腔（有参考）</div>
      <div class="cite-desc">台词 <code>わあ、すごい！とっても素敵ですね！</code>。无标版被文本正面情绪带跑、听感偏兴奋；有标版真正演绎出强忍泪水、勉强欢笑的哭腔颤音。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/10_mismatch_happy2cry.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/10_mismatch_happy2cry.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">证据 3（多向解读）：#02 含泪微颤 &amp; #03 冷峻逼问（有参考）</div>
      <div class="cite-desc">同一中性句 <code>……本当に、それでいいんですか？</code>。无标版在 #03 仍偏温柔犹豫、几乎无怒意；有标版在 #02 出现真实哭腔颤抖，在 #03 语气明显收紧冷硬。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">无标 #03 冷峻</span></div><audio controls preload="none" src="/samples/full_nocap_1000/03_ambigA_angry.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">有标 #03 冷峻</span></div><audio controls preload="none" src="/samples/full_captioned_1000/03_ambigA_angry.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">证据 4（纯 Caption 无参考）：#20 含泪微颤 &amp; #25 哽咽道谢（--no-ref）</div>
      <div class="cite-desc">脱离参考音频后差距同样清晰：无标版虽已保留真乃音色，但情绪易飘回平淡明朗；有标版全程保持吸气哽咽与微颤哭腔。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">无标 #25 无参考哭腔</span></div><audio controls preload="none" src="/samples/full_nocap_1000/25_noref_crying.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">有标 #25 无参考哭腔</span></div><audio controls preload="none" src="/samples/full_captioned_1000/25_noref_crying.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">证据 5（句内情绪转折）：#14 低声不安 → 坚定宣言（有参考）</div>
      <div class="cite-desc">有标版前半句更虚弱贴近气声、后半句「輝きたいんです！」的音量与情感抬升落差更明显；无标版整体起伏偏平缓。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/14_shift_timid2firm.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/14_shift_timid2firm.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">对照样本（差距较小）：#28 戏中戏低沉反差（--no-ref）</div>
      <div class="cite-desc">台词 <code>血を頂いても、いいですか……？</code> 本身自带强氛围暗示，两版全量均能压低声线、放慢节奏完成表演，有标版仅在停顿与气息细节上略更自然。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/28_noref_dark_act.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/28_noref_dark_act.wav"></audio></div>
      </div>
    </div>
  </div>
</div>

<script>
const MODEL_ORDER = [
  { name: 'base_zero_shot', label: 'Base Zero-Shot', cls: 'm-base', defaultOn: true },
  { name: 'no_caption_lora_3000', label: '无 Caption LoRA (3000步)', cls: 'm-nocap', defaultOn: false },
  { name: 'captioned_step_0000500', label: '有 Caption LoRA (500步)', cls: 'm-cap', defaultOn: false },
  { name: 'captioned_step_0001000', label: '有 Caption LoRA (1000步)', cls: 'm-cap', defaultOn: true },
  { name: 'full_nocap_1000', label: '全量微调 无 Caption (1000步)', cls: 'm-nocap', defaultOn: true },
  { name: 'full_captioned_1000', label: '全量微调 有 Caption (1000步)', cls: 'm-cap', defaultOn: true },
];

const GT_MAP = {
  '21_gt_05542.wav': '/samples/refs/mano_05542.wav',
  '22_gt_05543.wav': '/samples/refs/mano_05543.wav',
};

const CAPTION_ZH_MAP = {
  '01_ambigA_gentle.wav': '用沉稳柔和的语气，一边关心对方一边轻声询问。',
  '02_ambigA_teary.wav': '强忍泪水、声音微微颤抖，以几乎快要消失的小声怯生生地询问。',
  '03_ambigA_angry.wav': '压抑着怒气、用冰冷强硬的口吻，语尾干脆利落地逼问。',
  '04_ambigA_surp.wav': '倒吸一口气、吃惊地稍快语速、带着冲劲反问。',
  '05_ambigA_dark_act.wav': '用低沉沉稳而略带妖艳的语调，放慢节奏、留出停顿试探着发问。',
  '06_ambigB_happy.wav': '用明亮轻快的口吻，发自内心开心且元气满满地道谢。',
  '07_ambigB_crying.wav': '泪水哽咽、话语断续，饱含深深谢意地用颤抖的声音轻声诉说。',
  '08_ambigB_whisper.wav': '略带害羞地夹杂气息，用克制的耳语声温柔地道谢。',
  '09_ambigB_exhausted.wav': '疲惫不堪、抽干力气的平淡语调，勉强装作开朗般淡淡地说出。',
  '10_mismatch_happy2cry.wav': '强忍泪水、无力颤抖的声音，勉强装作开心般痛苦地说话。',
  '11_mismatch_sad2playful.wav': '故意用轻松明亮的调子，语速稍快地开玩笑般搪塞过去。',
  '12_mismatch_calm2panic.wav': '气喘吁吁、慌张急促地，用较快语速透出焦急地拼命传达。',
  '13_mismatch_soft2shout.wav': '仿佛抗拒般粗暴地拉高嗓音，拼命强烈地否定。',
  '14_shift_timid2firm.wav': '前半句没自信地用克制小声低语，后半句陡然一转、用力笔直而坚定地宣告。',
  '15_shift_laugh2warm.wav': '漏出轻柔的笑声后，满怀怀念地稍慢语速、温暖地娓娓道来。',
  '16_long_stage.wav': '一字一句清晰认真地停顿断句，用凛然饱满的语气饱含心意地倾诉。',
  '17_extreme_whisper.wav': '体贴对方般用极小的耳语声，夹杂大量气声悄悄说话。',
  '18_shortref_happy.wav': '用明亮轻快的语气，开心地询问。',
  '19_noref_happy.wav': '用明亮轻快的语气，开心地询问。',
  '20_noref_teary.wav': '强忍泪水、声音微微颤抖，怯生生且克制地询问。',
  '23_noref_angry.wav': '压抑着怒气、用冰冷强硬的口吻，语尾干脆利落地逼问。',
  '24_noref_whisper.wav': '体贴对方般用极小的耳语声，夹杂大量气声悄悄说话。',
  '25_noref_crying.wav': '泪水哽咽、话语断续，饱含深深谢意地用颤抖的声音轻声诉说。',
  '26_noref_shout.wav': '仿佛抗拒般粗暴地拉高嗓音，拼命强烈地否定。',
  '27_noref_shift.wav': '前半句没自信地用克制小声低语，后半句陡然一转、用力笔直而坚定地宣告。',
  '28_noref_dark_act.wav': '用低沉沉稳而略带妖艳的语调，放慢节奏、留出停顿试探着发问。',
  '29_noref_kw_cry.wav': '【极简短词】一边哭着、颤抖的声音',
  '30_noref_kw_angry.wav': '【极简短词】非常生气、强硬的语气',
  '31_noref_kw_sleepy.wav': '【极简短词】犯困般、迷迷糊糊、慢慢地',
  '32_noref_kw_fast_panic.wav': '【极简短词】焦急语速快、慌乱失措',
  '33_noref_tsundere.wav': '带着害羞掩饰又傲娇强硬的口吻，语速偏快地辩解。',
  '34_noref_scorn.wav': '无语嘲讽般冷冷哼笑，用居高临下轻蔑的口吻说话。',
  '35_noref_terror.wav': '因恐惧发抖、气息颤栗，以近乎快哭出来的紧迫声音喊叫。',
  '36_noref_cheer_loud.wav': '面向宽阔会场放开嗓门，全力明亮元气满满地高声呼喊。',
  '37_noref_mismatch_happy2dead.wav': '【无参考对抗】完全失去感情般棒读，用低沉冰冷平淡的语调说出开心台词。',
  '38_noref_mismatch_sad2laugh.wav': '【无参考对抗】憋不住快笑出声般，用欢乐开朗的调子说出绝望台词。',
  '21_gt_05542.wav': '低落失去自信般，用安静无力的声音喃喃自语。',
  '22_gt_05543.wav': '猛然回神后，用明亮积极、鼓励对方的轻快口吻说话。',
};

let evalGroups = [];
let enabledModels = new Set(MODEL_ORDER.filter(m => m.defaultOn).map(m => m.name));
let activeCategory = 'ALL';
let lastSampleKeys = '';

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function drawChart(svgId, seriesList) {
  const svg = document.getElementById(svgId);
  const W = 560, H = 200, padL = 48, padR = 14, padT = 16, padB = 24;
  const allPts = seriesList.flatMap(s => s.pts);
  if (allPts.length === 0) {
    svg.innerHTML = '<text x="20" y="30" fill="#94a3b8" font-size="13">暂无数据点</text>';
    return;
  }
  const minX = Math.min(...allPts.map(p => p.x));
  const maxX = Math.max(...allPts.map(p => p.x), minX + 1);
  let minY = Math.min(...allPts.map(p => p.y));
  let maxY = Math.max(...allPts.map(p => p.y));
  if (maxY - minY < 1e-6) { maxY += 0.1; minY -= 0.1; }
  const sx = x => padL + ((x - minX) / (maxX - minX)) * (W - padL - padR);
  const sy = y => H - padB - ((y - minY) / (maxY - minY)) * (H - padT - padB);

  let html = '';
  for (let i = 0; i <= 4; i++) {
    const yv = minY + (maxY - minY) * (i / 4);
    const yp = sy(yv);
    html += `<line x1="${padL}" y1="${yp}" x2="${W-padR}" y2="${yp}" stroke="#e2e8f0" stroke-dasharray="3,3"/>`;
    html += `<text x="${padL-6}" y="${yp+4}" text-anchor="end" font-size="10" fill="#64748b">${yv.toFixed(3)}</text>`;
  }
  html += `<text x="${padL}" y="${H-6}" font-size="10" fill="#64748b">step ${minX}</text>`;
  html += `<text x="${W-padR}" y="${H-6}" text-anchor="end" font-size="10" fill="#64748b">step ${maxX}</text>`;

  seriesList.forEach((s, idx) => {
    if (s.pts.length > 1) {
      const d = s.pts.map((p, i) => `${i===0?'M':'L'}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join(' ');
      html += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2"/>`;
    }
    if (s.dots) {
      s.pts.forEach(p => {
        html += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="3.5" fill="${s.color}"/>`;
      });
    }
    const last = s.pts[s.pts.length - 1];
    html += `<text x="${padL + 10 + idx*135}" y="13" font-size="11" fill="${s.color}" font-weight="600">${s.label}: ${last ? last.y.toFixed(4) : '-'}</text>`;
  });
  svg.innerHTML = html;
}

function orderedAvailableModels() {
  const byName = new Map(evalGroups.map(g => [g.name, g]));
  const result = [];
  for (const m of MODEL_ORDER) {
    if (byName.has(m.name)) result.push({ ...m, group: byName.get(m.name) });
  }
  for (const g of evalGroups) {
    if (!MODEL_ORDER.some(m => m.name === g.name)) {
      result.push({ name: g.name, label: g.name, cls: 'm-cap', defaultOn: true, group: g });
    }
  }
  return result;
}

function renderMatrix() {
  const avail = orderedAvailableModels();
  const togglesEl = document.getElementById('modelToggles');
  if (avail.length === 0) {
    togglesEl.innerHTML = '暂无已生成的对比模型';
    document.getElementById('samplesArea').innerHTML = '<div class="meta">等待首个模型生成完成...</div>';
    return;
  }

  // Render model toggle checkboxes
  togglesEl.innerHTML = '<span style="color:var(--muted)">显示模型列：</span>' + avail.map(m => {
    const checked = enabledModels.has(m.name) ? 'checked' : '';
    return `<label><input type="checkbox" ${checked} onchange="toggleModel('${esc(m.name)}', this.checked)"><span class="badge ${m.cls}">${esc(m.label)}</span></label>`;
  }).join('');

  // Build prompt list from first available group
  const baseItems = avail[0].group.items || [];
  const cats = ['ALL', ...Array.from(new Set(baseItems.map(it => it.category || '-')))];
  document.getElementById('catPills').innerHTML = cats.map(c => {
    const label = c === 'ALL' ? `全部 (${baseItems.length})` : c;
    const act = c === activeCategory ? 'active' : '';
    return `<button class="pill ${act}" onclick="setCategory('${esc(c)}')">${esc(label)}</button>`;
  }).join('');

  const cols = avail.filter(m => enabledModels.has(m.name));
  if (cols.length === 0) {
    document.getElementById('samplesArea').innerHTML = '<div class="meta">请至少勾选 1 个模型列。</div>';
    return;
  }

  // Map filename -> item per model
  const lookup = new Map();
  for (const m of cols) {
    const mMap = new Map();
    for (const it of (m.group.items || [])) {
      const fname = it.file.split('/').pop();
      mMap.set(fname, it);
    }
    lookup.set(m.name, mMap);
  }

  const rows = baseItems.filter(it => activeCategory === 'ALL' || (it.category || '-') === activeCategory);
  let html = '<table><colgroup><col class="prompt-col">' + cols.map(() => '<col class="model-col">').join('') + '</colgroup>';
  html += '<thead><tr><th>测试句 &amp; Caption 条件</th>' + cols.map(m => `<th><span class="badge ${m.cls}">${esc(m.label)}</span></th>`).join('') + '</tr></thead><tbody>';

  rows.forEach((it, idx) => {
    const fname = it.file.split('/').pop();
    const cat = it.category || '-';
    let mode = '<span class="badge">3条参考</span>';
    if (it.ref_mode === 'short') mode = '<span class="badge short">单条短参考</span>';
    if (it.ref_mode === 'none' || it.use_ref === false) mode = '<span class="badge none">无参考 (--no-ref)</span>';
    const gtUrl = GT_MAP[fname];
    const gtHtml = gtUrl ? `<div class="gt-box"><div class="meta"><b>真乃原声 GT：</b></div><audio controls preload="none" src="${gtUrl}"></audio></div>` : '';

    const capZh = CAPTION_ZH_MAP[fname] || '';
    const capZhHtml = capZh ? `<div class="prompt-cap-zh">译：${esc(capZh)}</div>` : '';

    html += `<tr><td>
      <div><span class="badge" style="background:#f1f5f9;color:#334155">#${String(idx+1).padStart(2,'0')}</span><span class="badge">${esc(cat)}</span>${mode}</div>
      <div class="prompt-text">${esc(it.text)}</div>
      <div class="prompt-cap">${esc(it.caption || '（无 caption）')}${capZhHtml}</div>
      ${gtHtml}
    </td>`;

    for (const m of cols) {
      const mIt = lookup.get(m.name).get(fname);
      if (mIt) {
        html += `<td><audio controls preload="none" src="/samples/${esc(mIt.file)}"></audio></td>`;
      } else {
        html += `<td class="meta">未生成</td>`;
      }
    }
    html += '</tr>';
  });

  html += '</tbody></table>';
  document.getElementById('samplesArea').innerHTML = html;

  // Auto-pause other audio players when one starts playing
  document.querySelectorAll('audio').forEach(a => {
    if (a.dataset.boundExclusive) return;
    a.dataset.boundExclusive = '1';
    a.addEventListener('play', () => {
      document.querySelectorAll('audio').forEach(other => {
        if (other !== a && !other.paused) other.pause();
      });
    });
  });
}

function toggleModel(name, checked) {
  if (checked) enabledModels.add(name); else enabledModels.delete(name);
  renderMatrix();
}

function setCategory(cat) {
  activeCategory = cat;
  renderMatrix();
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const prog = data.progress;
    if (prog) {
      document.getElementById('progressBar').style.width = prog.pct + '%';
      document.getElementById('progressText').textContent =
        `训练进度: ${prog.step} / ${prog.total} (${prog.pct}%) | ${prog.timing}`;
    }
    document.getElementById('gpuText').textContent = 'GPU: ' + data.gpu;

    const badge = document.getElementById('evalBadge');
    if (data.evaluating) {
      badge.style.display = 'inline-block';
      badge.textContent = '正在生成试听样本: ' + data.evaluating;
    } else {
      badge.style.display = 'none';
    }

    drawChart('chartLoss', [
      { label: 'train_loss', color: '#2563eb', pts: data.train.map(p => ({x: p.step, y: p.loss})) },
      { label: 'train_rf', color: '#0ea5e9', pts: data.train.map(p => ({x: p.step, y: p.rf})) },
      { label: 'valid_loss', color: '#dc2626', dots: true, pts: data.valid.map(p => ({x: p.step, y: p.loss})) },
    ]);
    drawChart('chartDur', [
      { label: 'dur_loss', color: '#16a34a', pts: data.train.map(p => ({x: p.step, y: p.dur})) },
      { label: 'valid_dur', color: '#ea580c', dots: true, pts: data.valid.map(p => ({x: p.step, y: p.dur})) },
    ]);

    const keys = JSON.stringify(data.eval_groups.map(g => `${g.name}:${(g.items||[]).length}`));
    if (keys !== lastSampleKeys) {
      lastSampleKeys = keys;
      evalGroups = data.eval_groups;
      renderMatrix();
    }
  } catch (e) {
    console.error(e);
  }
}

refresh();
setInterval(refresh, 10000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            log_data = parse_log()
            with _eval_status_lock:
                evaluating = _currently_evaluating
            payload = {
                **log_data,
                "gpu": get_gpu_info(),
                "evaluating": evaluating,
                "eval_groups": list_eval_groups(),
            }
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/samples/"):
            rel = unquote(self.path[len("/samples/"):]).lstrip("/")
            file_path = (EVAL_DIR / rel).resolve()
            if not str(file_path).startswith(str(EVAL_DIR.resolve())) or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = file_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def main():
    t = threading.Thread(target=evaluator_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    print("Monitor server running at http://0.0.0.0:8765/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
