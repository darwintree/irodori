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
    "/home/darwinwsl/.gemini/antigravity-cli/brain/a5bb3a90-52f7-4bab-990b-6ccbe32ba6b7/.system_generated/tasks/task-328.log"
)
OUTPUT_DIR = Path("/home/darwinwsl/code/irodori/outputs/mano_lora_baseline")
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
        "落ち込んで自信をなくしたように、静かで力のない声で呟く。",
    ),
    (
        "mano_05543.wav",
        "……！　うん、頑張ろうね、灯織ちゃん！",
        "ハッとした後、明るく前向きに元気づけるように話す。",
    ),
]

AMBIG_A = "……本当に、それでいいんですか？"
AMBIG_B = "プロデューサーさん、ありがとうございます。"

# (filename, category, text, caption, ref_mode: "full" | "short" | "none")
TEST_PROMPTS = [
    ("01_ambigA_gentle.wav", "多向解读 A", AMBIG_A, "落ち着いた優しい声で、相手を気遣いながら静かに尋ねる。", "full"),
    ("02_ambigA_timid.wav", "多向解读 A", AMBIG_A, "自信がなさそうに声を震わせながら、おずおずと控えめに聞く。", "full"),
    ("03_ambigA_angry.wav", "多向解读 A", AMBIG_A, "怒りを抑えたような冷たく強い口調で、きっぱりと問い詰める。", "full"),
    ("04_ambigA_surp.wav", "多向解读 A", AMBIG_A, "目を丸くして驚いたように、少し高めの声で勢いよく聞き返す。", "full"),
    ("05_ambigB_happy.wav", "多向解读 B", AMBIG_B, "明るく弾むような声で、心から嬉しそうに元気よくお礼を言う。", "full"),
    ("06_ambigB_teary.wav", "多向解读 B", AMBIG_B, "涙ぐみながら声を少し詰まらせて、感謝の気持ちを込めて静かに言う。", "full"),
    ("07_ambigB_whisper.wav", "多向解读 B", AMBIG_B, "少し照れたように息遣いの混じった小さな囁き声で、優しくお礼を言う。", "full"),
    ("08_mismatch_pos2neg.wav", "情绪不匹配对抗", "わあ、すごい！とっても素敵ですね！", "疲れ切ったように気だるげで、無理に笑っているような力のない静かなトーンで話す。", "full"),
    ("09_mismatch_neg2pos.wav", "情绪不匹配对抗", "ごめんなさい……私のせいで、全部台無しに……", "悲しみを隠すようにあえて明るく軽い調子で、早口にごまかすように話す。", "full"),
    ("10_long_complex.wav", "长句与读音稳定", "今日のステージは、今までみんなで練習してきた集大成だから、最後まで笑顔で精一杯届けたいと思います。イルミネーションスターズの輝きを、見ていてくださいね。", "落ち着いて凛とした張りのある声で、想いを込めて丁寧に語りかける。", "full"),
    ("11_shortref_happy.wav", "参考压力测试", AMBIG_A, "明るく弾むような声で、楽しそうに尋ねる。", "short"),
    ("12_noref_happy.wav", "参考压力测试", AMBIG_A, "若い女性の声。明るく弾むような声で、楽しそうに尋ねる。", "none"),
    ("13_noref_timid.wav", "参考压力测试", AMBIG_A, "若い女性の声。自信がなさそうに声を震わせながら、おずおずと控えめに聞く。", "none"),
    ("14_gt_05542.wav", "GT 原声对照", GT_SAMPLES[0][1], GT_SAMPLES[0][2], "full"),
    ("15_gt_05543.wav", "GT 原声对照", GT_SAMPLES[1][1], GT_SAMPLES[1][2], "full"),
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


def run_eval_for(name: str, step: int, lora_dir: Path | None) -> None:
    global _currently_evaluating
    target_dir = EVAL_DIR / name
    done_file = target_dir / "done.json"
    if done_file.exists():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    with _eval_status_lock:
        _currently_evaluating = name

    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["TOKENIZERS_PARALLELISM"] = "false"

    items = []
    try:
        for fname, category, prompt_text, prompt_caption, ref_mode in TEST_PROMPTS:
            out_wav = target_dir / fname
            cmd = [
                str(PYTHON_BIN),
                "infer.py",
                "--checkpoint", str(BASE_CKPT),
                "--model-precision", "bf16",
                "--codec-precision", "bf16",
                "--text", prompt_text,
                "--output-wav", str(out_wav),
                "--seed", "42",
                "--no-show-timings",
            ]
            if ref_mode == "full":
                cmd.extend(["--ref-wavs", *[str(p) for p in REF_WAVS]])
            elif ref_mode == "short":
                cmd.extend(["--ref-wav", str(SHORT_REF_WAV)])
            else:
                cmd.append("--no-ref")
            if prompt_caption:
                cmd.extend(["--caption", prompt_caption])
            if lora_dir is not None:
                cmd.extend(["--lora-adapter", str(lora_dir)])
            subprocess.run(cmd, cwd=str(IRODORI_REPO), env=env, check=True)
            items.append({
                "file": f"{name}/{fname}",
                "category": category,
                "text": prompt_text,
                "caption": prompt_caption,
                "ref_mode": ref_mode,
            })
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

    # First generate base model zero-shot baseline
    run_eval_for("base_zero_shot", 0, None)

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
                    # Extract step number (ignore decimal loss suffix on checkpoint_best_val_loss_0000500_0.758306)
                    m = re.match(r"checkpoint_(?:best_val_loss_)?(\d+)", ckpt.name)
                    if m:
                        step = int(m.group(1))
                        eval_name = f"step_{step:07d}"
                    elif ckpt.name == "checkpoint_final":
                        step = 999999
                        eval_name = "checkpoint_final"
                    else:
                        continue
                    if not (EVAL_DIR / eval_name / "done.json").exists():
                        time.sleep(2)
                        run_eval_for(eval_name, step, ckpt)
        except Exception as exc:
            print(f"[watcher] error: {exc}", flush=True)
        time.sleep(15)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Irodori-TTS 真乃 LoRA 训练监控台</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #f8fafc; color: #1e293b; }
  .card { background: #fff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
  h2, h3 { margin: 0 0 10px 0; }
  .meta { font-size: 13px; color: #475569; margin-bottom: 8px; }
  .bar-wrap { background: #e2e8f0; border-radius: 999px; height: 12px; overflow: hidden; margin: 8px 0; }
  .bar-fill { background: #3b82f6; height: 100%; transition: width .3s; }
  svg { width: 100%; height: 220px; background: #fcfcfd; border: 1px solid #f1f5f9; border-radius: 6px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  th, td { border: 1px solid #e2e8f0; padding: 8px 10px; font-size: 13px; text-align: left; }
  th { background: #f1f5f9; }
  audio { height: 32px; width: 260px; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; background: #dbeafe; color: #1d4ed8; }
</style>
</head>
<body>
<div class="card">
  <h2>Irodori-TTS 真乃 LoRA 训练监控台 <span id="evalBadge" class="badge" style="display:none"></span></h2>
  <div class="meta" id="progressText">加载中...</div>
  <div class="bar-wrap"><div class="bar-fill" id="progressBar" style="width:0%"></div></div>
  <div class="meta" id="gpuText">GPU: 加载中...</div>
</div>

<div class="row">
  <div class="card">
    <h3>Total Loss &amp; RF Loss (Train / Valid)</h3>
    <svg id="chartLoss" viewBox="0 0 560 220"></svg>
  </div>
  <div class="card">
    <h3>Duration Loss &amp; MAE (frames)</h3>
    <svg id="chartDur" viewBox="0 0 560 220"></svg>
  </div>
</div>

<div class="card">
  <h3>参考音频 &amp; 验证集原声对照（真乃原声）</h3>
  <div style="display:flex; gap:16px; flex-wrap:wrap;">
    <div><div class="meta">参考1: mano_00000.wav</div><audio controls preload="none" src="/samples/refs/mano_00000.wav"></audio></div>
    <div><div class="meta">参考2: mano_00002.wav</div><audio controls preload="none" src="/samples/refs/mano_00002.wav"></audio></div>
    <div><div class="meta">参考3: mano_00003.wav</div><audio controls preload="none" src="/samples/refs/mano_00003.wav"></audio></div>
    <div><div class="meta">GT1 (低落): mano_05542.wav</div><audio controls preload="none" src="/samples/refs/mano_05542.wav"></audio></div>
    <div><div class="meta">GT2 (元气): mano_05543.wav</div><audio controls preload="none" src="/samples/refs/mano_05543.wav"></audio></div>
  </div>
</div>

<div class="card">
  <h3>Checkpoint 示例生成对比试听（每 500 步自动生成）</h3>
  <div id="samplesArea" class="meta">等待首个样本生成...</div>
</div>

<script>
function drawChart(svgId, seriesList) {
  const svg = document.getElementById(svgId);
  const W = 560, H = 220, padL = 48, padR = 14, padT = 14, padB = 26;
  const allPts = seriesList.flatMap(s => s.pts);
  if (allPts.length === 0) {
    svg.innerHTML = '<text x="20" y="30" fill="#94a3b8" font-size="13">暂无数据点（每 50 步记录一次）</text>';
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
    html += `<text x="${padL + 10 + idx*135}" y="14" font-size="11" fill="${s.color}" font-weight="600">${s.label}: ${last ? last.y.toFixed(4) : '-'}</text>`;
  });
  svg.innerHTML = html;
}

let lastSampleKeys = '';

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const prog = data.progress;
    if (prog) {
      document.getElementById('progressBar').style.width = prog.pct + '%';
      document.getElementById('progressText').textContent =
        `进度: ${prog.step} / ${prog.total} (${prog.pct}%) | ${prog.timing}`;
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

    const keys = JSON.stringify(data.eval_groups.map(g => g.name));
    if (keys !== lastSampleKeys) {
      lastSampleKeys = keys;
      if (data.eval_groups.length === 0) {
        document.getElementById('samplesArea').innerHTML = '<div class="meta">暂无已生成的样本</div>';
      } else {
        let html = '';
        for (const g of data.eval_groups) {
          html += `<h4>${g.name} (step ${g.step})</h4><table><thead><tr><th>测试维度</th><th>参考模式</th><th>测试台词</th><th>感情控制 (Caption)</th><th>生成音频</th></tr></thead><tbody>`;
          for (const it of g.items) {
            const cat = it.category || '-';
            let mode = '<span class="badge">3条参考</span>';
            if (it.ref_mode === 'short') mode = '<span class="badge" style="background:#e0e7ff;color:#3730a3">单条短参考</span>';
            if (it.ref_mode === 'none' || it.use_ref === false) mode = '<span class="badge" style="background:#fef3c7;color:#92400e">无参考 (--no-ref)</span>';
            const cap = it.caption ? it.caption : '<span style="color:#94a3b8">（无 caption）</span>';
            html += `<tr><td><b>${cat}</b></td><td>${mode}</td><td>${it.text}</td><td>${cap}</td><td><audio controls preload="none" src="/samples/${it.file}"></audio></td></tr>`;
          }
          html += '</tbody></table><br>';
        }
        document.getElementById('samplesArea').innerHTML = html;
      }
    }
  } catch (e) {
    console.error(e);
  }
}

refresh();
setInterval(refresh, 8000);
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
