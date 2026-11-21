#!/usr/bin/env python3
"""Focused 40-item --no-ref (caption-only) cross-model benchmark server on port 8765.

Evaluates 20 Multi-directional prompts (4 neutral texts x 5 directions) + 20 Extreme Emotion prompts
across Base Zero-Shot, Full FT No-Caption (1000 steps), and Full FT Captioned (1000 steps).
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

EVAL_DIR = Path("/home/darwinwsl/code/irodori/outputs/mano_full_ab_noref_eval")
HUMAN_VOTES_FILE = EVAL_DIR / "human_votes.json"
BASE_CKPT = Path("/home/darwinwsl/code/irodori/models/Irodori-TTS-v4.1-Small/model.safetensors")
FULL_NOCAP_CKPT = Path("/home/darwinwsl/code/irodori/outputs/mano_full_nocap/checkpoint_best_val_loss_0001000_0.728665.pt")
FULL_CAP_CKPT = Path("/home/darwinwsl/code/irodori/outputs/mano_full_captioned/checkpoint_best_val_loss_0001000_0.710021.pt")
IRODORI_REPO = Path("/home/darwinwsl/code/irodori/Irodori-TTS")
PYTHON_BIN = IRODORI_REPO / ".venv/bin/python"
BATCH_INFER = Path("/home/darwinwsl/code/irodori/scripts/batch_infer_eval.py")

TEXT_A = "……本当に、それでいいんですか？"
TEXT_B = "プロデューサーさん、ありがとうございます。"
TEXT_C = "あの……今、少しお時間よろしいですか？"
TEXT_D = "わかりました。私に任せてください。"

# (filename, category, group_tag, text, caption_ja, caption_zh)
PROMPTS: list[tuple[str, str, str, str, str, str]] = [
    # === 1. 多种解读方向 (20条: 4句 × 5方向) ===
    ("01_A_gentle.wav", "多种解读方向", "句A", TEXT_A, "落ち着いた柔らかな口調で、相手を気遣いながら静かに尋ねている。", "沉稳柔和，一边关心对方一边轻声询问"),
    ("02_A_teary.wav", "多种解读方向", "句A", TEXT_A, "涙を堪えながら声を小さく震わせ、今にも消えそうな音量でおずおずと尋ねている。", "强忍泪水、声音微颤，以近乎消失的小声怯生生询问"),
    ("03_A_angry.wav", "多种解读方向", "句A", TEXT_A, "怒りを抑えた冷たく強い口調で、語尾を言い切るようにきっぱりと問い詰めている。", "压抑怒气、冰冷强硬，语尾干脆利落地逼问"),
    ("04_A_surp.wav", "多种解读方向", "句A", TEXT_A, "息を呑んで驚いたように、やや早口で勢いよく聞き返している。", "倒吸一口气、吃惊地稍快语速带冲劲反问"),
    ("05_A_happy.wav", "多种解读方向", "句A", TEXT_A, "明るく弾んだ口調で、嬉しそうに期待を込めて楽しげに尋ねている。", "明亮轻快，满怀开心与期待地愉快询问"),

    ("06_B_happy.wav", "多种解读方向", "句B", TEXT_B, "明るく弾んだ口調で、心から嬉しそうに元気よくお礼を伝えている。", "明亮轻快，发自内心开心元气满满地道谢"),
    ("07_B_crying.wav", "多种解读方向", "句B", TEXT_B, "涙に詰まって言葉を途切れさせながら、深い感謝を込めて震える声で静かに伝えている。", "泪水哽咽、话语断续，饱含谢意颤声轻说"),
    ("08_B_whisper.wav", "多种解读方向", "句B", TEXT_B, "少し照れくさそうに息を漏らしながら、控えめな囁き声で優しくお礼を言っている。", "略带害羞夹杂气息，克制耳语温柔道谢"),
    ("09_B_tired.wav", "多种解读方向", "句B", TEXT_B, "疲れ切ったように力を抜いた低いトーンで、ゆっくりとか細い声で静かにお礼を言っている。", "疲惫抽干力气的低沉语气，缓慢微弱静静道谢"),
    ("10_B_formal.wav", "多种解读方向", "句B", TEXT_B, "凛とした張りのある口調で、一語一語をはっきりと丁寧に感謝を伝えている。", "凛然饱满的语气，一字一句清晰郑重地致谢"),

    ("11_C_timid.wav", "多种解读方向", "句C", TEXT_C, "自信がなさそうに言葉を詰まらせながら、控えめで弱々しい小声でおずおずと声をかけている。", "没自信、吞吞吐吐，克制微弱小声怯生生搭话"),
    ("12_C_panic.wav", "多种解读方向", "句C", TEXT_C, "息を切らして焦ったように、早口で切迫感を滲ませながら呼び止めている。", "气喘吁吁、焦急早口带紧迫感地叫住对方"),
    ("13_C_casual.wav", "多种解读方向", "句C", TEXT_C, "親しい相手に対して、柔らかくくだけた明るい口調で気さくに話しかけている。", "面对亲近之人，柔和放松明亮随和地搭话"),
    ("14_C_serious.wav", "多种解读方向", "句C", TEXT_C, "真剣な空気をまとった落ち着いた低い声で、ゆっくりと間を取りながら切り出している。", "严肃沉稳偏低的声音，放慢停顿郑重开口"),
    ("15_C_shy.wav", "多种解读方向", "句C", TEXT_C, "少し照れながら甘えるような柔らかな語尾で、遠慮がちにそっと話しかけている。", "略带害羞、语尾柔和带点撒娇，客气轻声搭话"),

    ("16_D_firm.wav", "多种解读方向", "句D", TEXT_D, "迷いのない力強い口調で、はっきりと言い切るように頼もしく答えている。", "毫不犹豫、有力干脆且可靠地回答"),
    ("17_D_forced.wav", "多种解读方向", "句D", TEXT_D, "不安を隠すように少し声を震わせながら、無理に気丈に振る舞うような硬い口調で答えている。", "掩饰不安、声音微颤，勉强硬撑着僵硬回答"),
    ("18_D_cheerful.wav", "多种解读方向", "句D", TEXT_D, "軽やかで明るいテンポで、楽しそうに弾んだ声で快く引き受けている。", "轻快明亮节奏，开心弹跳地爽快答应"),
    ("19_D_cold.wav", "多种解读方向", "句D", TEXT_D, "感情を抑えた冷たく乾いた口調で、事務的にそっけなく返事をしている。", "压抑感情、冰冷干涩，公事公办地冷淡回应"),
    ("20_D_soothing.wav", "多种解读方向", "句D", TEXT_D, "相手を安心させるように優しく包み込む柔らかな声で、穏やかにゆったりと答えている。", "安抚对方般温柔包容，平稳舒缓地回答"),

    # === 2. 极端情绪 (20条) ===
    ("21_ext_breakdown.wav", "极端情绪", "哭泣崩溃", "だって……こうでもしなかったら……私……", "嗚咽を堪えながら声を詰まらせ、今にも消えそうな弱い声で途切れ途切れに漏らしている。", "强忍呜咽哽咽断续，近乎消失的微弱哭声"),
    ("22_ext_sob_apology.wav", "极端情绪", "哭泣崩溃", "ごめんなさい……私のせいで、みんなに迷惑をかけてしまって……", "涙声で息を吸い込みながら、深く自責するように震える声で必死に謝っている。", "带哭腔吸气，深感自责地颤声拼命道歉"),
    ("23_ext_helpless.wav", "极端情绪", "哭泣崩溃", "もう、どうしていいかわからないんです……っ", "感情が溢れて今にも泣き出しそうな声で、語尾を震わせながら弱々しく訴えている。", "情绪溢满快哭出来，语尾发颤无助倾诉"),
    ("24_ext_hollow.wav", "极端情绪", "哭泣崩溃", "……そっか。もう、会えないんですね……", "力を完全に失ったか細い声で、諦めと寂しさを滲ませながらゆっくりと呟いている。", "彻底失去力气的微弱空洞声音，绝望落寞低语"),

    ("25_ext_protest.wav", "极端情绪", "愤怒爆发", "やめてくださいっ！そんな言い方、ひどすぎます！", "感情を昂ぶらせて声を荒らげ、強い怒りと抗議を込めて勢いよく叫んでいる。", "情绪激动拉高嗓音，带着强烈怒气与抗议喊出"),
    ("26_ext_reject.wav", "极端情绪", "愤怒爆发", "私、そんなこと望んでいません……！", "拒絶するように声を張り上げ、苦しげに力を込めて強く否定している。", "抗拒般放声高喊，痛苦用力强烈否定"),
    ("27_ext_snap.wav", "极端情绪", "愤怒爆发", "もういいです！勝手にすればいいじゃないですか！", "我慢の限界に達したように苛立ちを爆発させ、強い口調で突き放すように言い捨てている。", "忍无可忍爆发烦躁，强硬甩话怒怼"),
    ("28_ext_scorn.wav", "极端情绪", "愤怒爆发", "へぇ……それ、本気で言ってるんですか？", "呆れて馬鹿にしたように、冷たく鼻で笑いながら見下す口調で話している。", "无语嘲讽冷冷哼笑，居高临下轻蔑反问"),

    ("29_ext_terror.wav", "极端情绪", "恐惧慌乱", "だ、誰か……そこにいるんですか……？来ないで……！", "恐怖に怯えて息を震わせながら、今にも泣き出しそうな切迫した声で叫んでいる。", "因恐惧发抖气息颤栗，近乎哭出来的紧迫喊叫"),
    ("30_ext_panic.wav", "极端情绪", "恐惧慌乱", "大変ですっ！もうすぐ本番なのに衣装が見つからなくて……！", "息を切らして慌てふためくように、非常に早いテンポで焦りを剥き出しにして話している。", "气喘吁吁极度慌乱，极快语速焦急倾诉"),
    ("31_ext_stutter.wav", "极端情绪", "恐惧慌乱", "あ、あのっ……っ、えっと……は、はじめまして……っ", "極度に緊張して息が詰まるように、言葉を何度も途切れさせながら小刻みに話している。", "极度紧张气短，多次断续结巴局促开口"),

    ("32_ext_whisper_bird.wav", "极端情绪", "耳语困倦", "……しっ、静かにしてください。あそこに小鳥さんが休んでいますよ……", "相手を気遣うように極小の囁き声で、息を多めに含ませながらそっと話している。", "体贴对方的极小耳语，夹杂大量气声悄悄说话"),
    ("33_ext_whisper_night.wav", "极端情绪", "耳语困倦", "……おやすみなさい。また明日、お会いしましょうね……", "静かな夜に寄り添うように、息遣い多めのとても優しく小さな囁き声でゆったりと語っている。", "极轻多气声的温柔睡前耳语，舒缓轻语"),
    ("34_ext_sleepy.wav", "极端情绪", "耳语困倦", "ふぁ……もう、こんな時間なんですね……少し、眠くなってきちゃいました……", "眠気に包まれてぼんやりとした声で、語尾を伸ばしながらとてもゆっくりと脱力して話している。", "困倦迷糊、语尾拉长、极其缓慢脱力地说话"),

    ("35_ext_live_cheer.wav", "极端情绪", "狂喜欢呼", "みんなーっ！今日は最後まで一緒に盛り上がっていこうねーっ！！", "広い会場に向かって声を張り上げ、全力で明るく元気いっぱいに呼びかけている。", "面向宽阔会场放开嗓门，全力明亮元气喊麦"),
    ("36_ext_ecstatic.wav", "极端情绪", "狂喜欢呼", "やったぁっ！本当に合格したんですね！夢みたいですっ！", "抑えきれない喜びを爆発させて、高く弾んだ声で勢いよく歓声を上げている。", "按捺不住狂喜爆发，高扬轻快欢呼"),
    ("37_ext_laughing.wav", "极端情绪", "狂喜欢呼", "あははっ！もう、プロデューサーさんったら、びっくりさせないでくださいよっ！", "楽しそうに明るく笑い声を上げながら、軽快で弾んだテンポで親しげに話している。", "开朗笑出声来，轻快弹跳亲切地说话"),

    ("38_ext_vow.wav", "极端情绪", "坚定反差", "絶対に負けません！私たちが、一番星になってみせますっ！", "揺るぎない決意を込めて声に強い張りを出し、熱くまっすぐに言い切っている。", "饱含坚定决心、声音饱满有力，炽热斩钉截铁宣言"),
    ("39_ext_vampire.wav", "极端情绪", "坚定反差", "血を頂いても、いいですか……？逃がしませんよ……", "低く落ち着いた妖艶なトーンで、ゆっくりと間を取りながら囁くように迫っている。", "低沉妖艳缓慢留白，贴近试探的戏中戏反差"),
    ("40_ext_shift.wav", "极端情绪", "坚定反差", "……私なんかがセンターでいいのかなって、ずっと不安でした。でも、今はみんなと一緒に輝きたいんですっ！", "前半は自信なさげに控えめな小声で呟き、後半は一転して力を込めてまっすぐ力強く宣言している。", "前半句没自信极弱低语，后半句陡然有力坚定宣言"),
]

MODELS = [
    ("base_zero_shot", "Base Zero-Shot", "m-base", BASE_CKPT, False),
    ("full_nocap_1000", "全量 无 Caption (1000步)", "m-nocap", FULL_NOCAP_CKPT, True),
    ("full_captioned_1000", "全量 有 Caption (1000步)", "m-cap", FULL_CAP_CKPT, True),
]

AI_BLIND_WINNER: dict[str, str] = {
    "01_A_gentle.wav": "Tie",
    "02_A_teary.wav": "full_captioned_1000",
    "03_A_angry.wav": "full_captioned_1000",
    "04_A_surp.wav": "Tie",
    "05_A_happy.wav": "full_nocap_1000",
    "06_B_happy.wav": "full_nocap_1000",
    "07_B_crying.wav": "full_captioned_1000",
    "08_B_whisper.wav": "full_nocap_1000",
    "09_B_tired.wav": "full_captioned_1000",
    "10_B_formal.wav": "full_captioned_1000",
    "11_C_timid.wav": "full_captioned_1000",
    "12_C_panic.wav": "full_captioned_1000",
    "13_C_casual.wav": "full_nocap_1000",
    "14_C_serious.wav": "full_captioned_1000",
    "15_C_shy.wav": "full_captioned_1000",
    "16_D_firm.wav": "full_captioned_1000",
    "17_D_forced.wav": "full_captioned_1000",
    "18_D_cheerful.wav": "Tie",
    "19_D_cold.wav": "full_nocap_1000",
    "20_D_soothing.wav": "full_captioned_1000",
    "21_ext_breakdown.wav": "full_captioned_1000",
    "22_ext_sob_apology.wav": "full_captioned_1000",
    "23_ext_helpless.wav": "full_captioned_1000",
    "24_ext_hollow.wav": "full_captioned_1000",
    "25_ext_protest.wav": "full_captioned_1000",
    "26_ext_reject.wav": "full_nocap_1000",
    "27_ext_snap.wav": "full_nocap_1000",
    "28_ext_scorn.wav": "full_captioned_1000",
    "29_ext_terror.wav": "full_captioned_1000",
    "30_ext_panic.wav": "full_captioned_1000",
    "31_ext_stutter.wav": "full_nocap_1000",
    "32_ext_whisper_bird.wav": "full_captioned_1000",
    "33_ext_whisper_night.wav": "full_nocap_1000",
    "34_ext_sleepy.wav": "Tie",
    "35_ext_live_cheer.wav": "full_captioned_1000",
    "36_ext_ecstatic.wav": "full_captioned_1000",
    "37_ext_laughing.wav": "full_nocap_1000",
    "38_ext_vow.wav": "full_captioned_1000",
    "39_ext_vampire.wav": "full_nocap_1000",
    "40_ext_shift.wav": "full_captioned_1000",
}

VALID_WINNERS = {"", "full_captioned_1000", "full_nocap_1000", "Tie"}
PROMPT_FNAMES = {p[0] for p in PROMPTS}

_lock = threading.Lock()
_votes_lock = threading.Lock()
_currently_evaluating: str | None = None


def load_human_votes() -> dict[str, dict[str, str]]:
    with _votes_lock:
        if not HUMAN_VOTES_FILE.is_file():
            return {}
        try:
            raw = json.loads(HUMAN_VOTES_FILE.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("votes"), dict):
                return raw["votes"]
            if isinstance(raw, dict):
                return raw
        except Exception:
            pass
        return {}


def save_human_votes(votes: dict[str, dict[str, str]]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "votes": votes,
    }
    tmp = HUMAN_VOTES_FILE.with_suffix(".json.tmp")
    with _votes_lock:
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, HUMAN_VOTES_FILE)


def build_export_payload(votes: dict[str, dict[str, str]]) -> dict:
    items = []
    cap_wins = 0
    nocap_wins = 0
    ties = 0
    agreed = 0
    compared = 0
    by_cat: dict[str, dict[str, int]] = {}

    for idx, (fname, cat, grp, text, cap_ja, cap_zh) in enumerate(PROMPTS, start=1):
        v = votes.get(fname, {})
        h_win = v.get("winner", "")
        h_note = v.get("note", "")
        h_ts = v.get("updated_at", "")
        ai_win = AI_BLIND_WINNER.get(fname, "")
        agree = None
        if h_win:
            compared += 1
            agree = (h_win == ai_win)
            if agree:
                agreed += 1
            by_cat.setdefault(cat, {"full_captioned_1000": 0, "full_nocap_1000": 0, "Tie": 0, "judged": 0})
            by_cat[cat]["judged"] += 1
            if h_win == "full_captioned_1000":
                cap_wins += 1
                by_cat[cat]["full_captioned_1000"] += 1
            elif h_win == "full_nocap_1000":
                nocap_wins += 1
                by_cat[cat]["full_nocap_1000"] += 1
            elif h_win == "Tie":
                ties += 1
                by_cat[cat]["Tie"] += 1
        items.append({
            "index": idx,
            "fname": fname,
            "category": cat,
            "group": grp,
            "text": text,
            "caption": cap_ja,
            "caption_zh": cap_zh,
            "ai_blind_winner": ai_win,
            "human_winner": h_win,
            "agree_with_ai": agree,
            "human_note": h_note,
            "updated_at": h_ts,
        })

    decisive = cap_wins + nocap_wins
    return {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_prompts": len(PROMPTS),
        "judged_count": compared,
        "summary": {
            "full_captioned_1000_wins": cap_wins,
            "full_nocap_1000_wins": nocap_wins,
            "ties": ties,
            "captioned_win_rate_excl_tie": round(cap_wins / decisive, 4) if decisive else None,
            "by_category": by_cat,
            "human_vs_ai_agreement": {
                "agreed": agreed,
                "disagreed": compared - agreed,
                "total_compared": compared,
                "agreement_rate": round(agreed / compared, 4) if compared else None,
            },
        },
        "items": items,
    }


def run_eval_model(name: str, ckpt: Path) -> None:
    global _currently_evaluating
    target_dir = EVAL_DIR / name
    done_file = target_dir / "done.json"
    if done_file.exists():
        try:
            ex = json.loads(done_file.read_text(encoding="utf-8"))
            if len(ex.get("items", [])) == len(PROMPTS):
                return
        except Exception:
            pass
    target_dir.mkdir(parents=True, exist_ok=True)
    with _lock:
        _currently_evaluating = name

    env = os.environ.copy()
    env["IRODORI_SDPA_CUDNN"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    jobs = []
    items = []
    for fname, cat, grp, text, cap_ja, cap_zh in PROMPTS:
        out_wav = target_dir / fname
        jobs.append({
            "out_wav": str(out_wav),
            "text": text,
            "caption": cap_ja,
            "ref_mode": "none",
        })
        items.append({
            "file": f"{name}/{fname}",
            "fname": fname,
            "category": cat,
            "group": grp,
            "text": text,
            "caption": cap_ja,
            "caption_zh": cap_zh,
        })

    try:
        if any(not Path(j["out_wav"]).exists() for j in jobs):
            jobs_file = target_dir / "_jobs.json"
            jobs_file.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
            subprocess.run(
                [str(PYTHON_BIN), str(BATCH_INFER), "--checkpoint", str(ckpt), "--jobs-json", str(jobs_file)],
                cwd=str(IRODORI_REPO),
                env=env,
                check=True,
            )
            jobs_file.unlink(missing_ok=True)
        done_file.write_text(
            json.dumps({"name": name, "items": items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"[eval] failed for {name}: {exc}", flush=True)
    finally:
        with _lock:
            _currently_evaluating = None


def worker_loop() -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # Generate Full FT A/B first so user can compare them fastest, then Base Zero-Shot
    for name, _, _, ckpt, _ in [MODELS[1], MODELS[2], MODELS[0]]:
        if ckpt.exists():
            run_eval_model(name, ckpt)


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Irodori-TTS 无参考仅 Caption 40 题跨模型对比台</title>
<style>
  :root { --bg: #f8fafc; --card: #fff; --border: #e2e8f0; --text: #0f172a; --muted: #475569; --soft: #f1f5f9; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 16px 20px 60px; background: var(--bg); color: var(--text); }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
  h2, h3 { margin: 0 0 8px 0; }
  .meta { font-size: 13px; color: var(--muted); }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 10px 0; }
  .pill { background: var(--soft); border: 1px solid var(--border); color: var(--muted); padding: 4px 10px; border-radius: 999px; cursor: pointer; font-size: 12px; }
  .pill.active { background: #dbeafe; border-color: #2563eb; color: #1d4ed8; font-weight: 600; }
  .model-toggles { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; padding: 8px 12px; background: var(--soft); border-radius: 8px; font-size: 13px; }
  .model-toggles label { display: inline-flex; align-items: center; gap: 5px; cursor: pointer; font-weight: 600; }
  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; table-layout: fixed; }
  th, td { border: 1px solid var(--border); padding: 10px; font-size: 13px; vertical-align: top; text-align: left; }
  th { background: var(--soft); position: sticky; top: 0; z-index: 2; }
  col.prompt-col { width: 360px; }
  col.model-col { width: 235px; }
  col.human-col { width: 235px; }
  .prompt-text { font-size: 14px; font-weight: 600; margin: 5px 0; line-height: 1.45; }
  .prompt-cap { font-size: 12.5px; color: #1e40af; background: #eff6ff; border-left: 3px solid #3b82f6; padding: 5px 8px; border-radius: 4px; line-height: 1.4; }
  .prompt-cap-zh { font-size: 12px; color: #475569; margin-top: 3px; }
  audio { height: 34px; width: 100%; min-width: 190px; }
  .badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px; font-weight: 600; background: #dbeafe; color: #1d4ed8; margin-right: 4px; }
  .badge.grp { background: #f1f5f9; color: #334155; }
  .badge.m-base { background: #e2e8f0; color: #334155; }
  .badge.m-nocap { background: #fee2e2; color: #991b1b; }
  .badge.m-cap { background: #dcfce7; color: #166534; }
  .badge.agree { background: #dcfce7; color: #166534; }
  .badge.disagree { background: #fef3c7; color: #92400e; }
  tr:hover td { background: #fafcff; }
  .human-bar { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 10px; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px; }
  .human-stats { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .human-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .btn { background: #fff; border: 1px solid #cbd5e1; color: var(--text); padding: 5px 11px; border-radius: 6px; cursor: pointer; font-size: 12.5px; font-weight: 600; text-decoration: none; display: inline-flex; align-items: center; gap: 4px; }
  .btn:hover { background: #f1f5f9; }
  .btn.primary { background: #2563eb; border-color: #2563eb; color: #fff; }
  .btn.primary:hover { background: #1d4ed8; }
  .btn.danger { color: #b91c1c; border-color: #fecaca; }
  .btn.danger:hover { background: #fef2f2; }
  .vote-btns { display: flex; gap: 5px; margin-bottom: 6px; }
  .vbtn { flex: 1; background: #fff; border: 1px solid #cbd5e1; color: #334155; padding: 5px 4px; border-radius: 6px; cursor: pointer; font-size: 12px; font-weight: 600; transition: all .12s; }
  .vbtn:hover { background: #f1f5f9; }
  .vbtn.active-cap { background: #16a34a; border-color: #16a34a; color: #fff; }
  .vbtn.active-nocap { background: #dc2626; border-color: #dc2626; color: #fff; }
  .vbtn.active-tie { background: #475569; border-color: #475569; color: #fff; }
  .note-input { width: 100%; border: 1px solid #cbd5e1; border-radius: 6px; padding: 5px 7px; font-size: 12px; font-family: inherit; resize: vertical; min-height: 44px; background: #fff; color: var(--text); }
  .note-input:focus { outline: none; border-color: #2563eb; box-shadow: 0 0 0 2px rgba(37,99,235,.12); }
  .vote-meta { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; min-height: 20px; }
  .save-hint { font-size: 11.5px; color: #64748b; }
  .verdict-card { border-left: 4px solid #16a34a; }
  .verdict-summary { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px 14px; font-size: 13.5px; line-height: 1.65; margin-bottom: 12px; }
  .verdict-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 12px; }
  .cite-box { background: #f8fafc; border: 1px solid var(--border); border-radius: 8px; padding: 10px 12px; }
  .cite-title { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
  .cite-desc { font-size: 12.5px; color: var(--muted); line-height: 1.45; margin-bottom: 8px; }
  .cite-pair { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
  .cite-pair audio { width: 100%; min-width: 140px; height: 32px; }
  td.win-cap { background: #f0fdf4; box-shadow: inset 0 0 0 2px #86efac; }
  td.win-nocap { background: #fef2f2; box-shadow: inset 0 0 0 2px #fca5a5; }
  td.win-tie { background: #f8fafc; }
  .win-tag { display: inline-block; margin-bottom: 6px; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }
  .win-tag.cap { background: #16a34a; color: #fff; }
  .win-tag.nocap { background: #dc2626; color: #fff; }
  .win-tag.tie { background: #cbd5e1; color: #334155; }
</style>
</head>
<body>
<div class="card">
  <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
    <h2>无参考仅 Caption（--no-ref）40 题跨模型对比台 <span id="evalBadge" class="badge" style="display:none"></span></h2>
    <div class="meta">20 题多种解读方向（4 句 × 5 向） + 20 题极端情绪 · 去除对抗场景 · 点击任一音频自动暂停其他</div>
  </div>
</div>

<div class="card">
  <div class="human-bar" id="humanBar">
    <div class="human-stats" id="humanStats">加载我的评判统计...</div>
    <div class="human-actions">
      <a class="btn primary" href="/api/export?format=json" download="human_eval_noref_40.json">导出 JSON</a>
      <a class="btn" href="/api/export?format=csv" download="human_eval_noref_40.csv">导出 CSV</a>
      <button class="btn" onclick="document.getElementById('importFile').click()">导入 JSON</button>
      <input type="file" id="importFile" accept=".json,application/json" style="display:none" onchange="importVotes(this)">
      <button class="btn danger" onclick="clearAllVotes()">清空全部</button>
    </div>
  </div>
  <div class="model-toggles" id="modelToggles">加载中...</div>
  <div class="controls" id="catPills"></div>
  <div class="controls" id="grpPills"></div>
  <div class="controls" id="judgePills"></div>
  <div class="controls" id="humanPills"></div>
  <div class="table-wrap" id="samplesArea"><div class="meta">等待样本生成...</div></div>
</div>

<div class="card verdict-card">
  <h3>AI 严格双盲实测结果（40 题纯无参考 --no-ref，每题独立随机打乱 A/B 顺序后盲评再揭晓）</h3>
  <div class="verdict-summary">
    <b>统计结论：有 Caption 版显著占优，但并非「全面碾压」，属于中等偏强的真实提升（双侧二项检验 <code>p = 0.0288 &lt; 0.05</code>）：</b><br>
    • <b>总计（40 题）</b>：<b>有 Caption 胜 25</b> ｜ <b>无 Caption 胜 11</b> ｜ <b>平局 4</b>（去平局净胜率 <b>69.4%</b>，<code>25/36</code>）<br>
    • <b>多种解读方向（20 题）</b>：有 Caption 胜 <b>12</b> ｜ 无 Caption 胜 <b>5</b> ｜ 平局 <b>3</b>（净胜率 <b>70.6%</b>）<br>
    • <b>极端情绪（20 题）</b>：有 Caption 胜 <b>13</b> ｜ 无 Caption 胜 <b>6</b> ｜ 平局 <b>1</b>（净胜率 <b>68.4%</b>）<br>
    • <b>规律拆解</b>：有标版在<b>哭腔/哽咽/失落（#02/#07/#21/#22/#23/#24 全胜 6/6）</b>、<b>疲惫/勉强硬撑/紧张慌乱（#09/#11/#12/#14/#17/#30 全胜）</b>以及<b>坚定/恐惧/转折（#10/#16/#29/#38/#40）</b>上高度稳定；而无标版在 <b>11 题</b>中胜出（<code>#05/#06/#08/#13/#19/#26/#27/#31/#33/#37/#39</code>），主要集中在真乃高频的明朗欢快/轻松日常与部分强文本引导句上，说明无标版在这些区间不仅未退化，有时听感还更松弛自然。
  </div>

  <div class="verdict-grid">
    <div class="cite-box">
      <div class="cite-title">多种解读 #17：句D 勉强硬撑（最典型差距）</div>
      <div class="cite-desc">台词 <code>わかりました。私に任せてください。</code>。无标版听起来元气满满；有标版听出明显的紧绷微颤与掩饰不安的硬撑感。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/17_D_forced.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/17_D_forced.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">多种解读 #09：句B 疲惫低落</div>
      <div class="cite-desc">无标版仍偏普通明亮道谢；有标版明显抽干力气，带出疲惫气声与低沉缓慢感。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/09_B_tired.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/09_B_tired.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">多种解读 #12：句C 慌张急促</div>
      <div class="cite-desc">有标版明显更气喘、语速更快且切迫感强；无标版偏温和试探。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/12_C_panic.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/12_C_panic.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">极端情绪 #21：哭泣崩溃</div>
      <div class="cite-desc">两版都有哭腔，但有标版的哽咽停顿、吸气颤音与脆弱感明显更真实痛切。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">全量 无 Caption</span></div><audio controls preload="none" src="/samples/full_nocap_1000/21_ext_breakdown.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">全量 有 Caption</span></div><audio controls preload="none" src="/samples/full_captioned_1000/21_ext_breakdown.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">极端情绪 #25：愤怒抗议 &amp; #29：极度恐惧</div>
      <div class="cite-desc">有标版在「やめてくださいっ！」的愤怒冲劲以及「来ないで！」的恐惧尖锐颤栗上，情绪爆发上限均高于无标版。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">无标 #29 恐惧</span></div><audio controls preload="none" src="/samples/full_nocap_1000/29_ext_terror.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">有标 #29 恐惧</span></div><audio controls preload="none" src="/samples/full_captioned_1000/29_ext_terror.wav"></audio></div>
      </div>
    </div>

    <div class="cite-box">
      <div class="cite-title">接近对照组 #19（冷淡疏离）&amp; #34（极度困倦）</div>
      <div class="cite-desc">#19 两版均表现干脆（无标版甚至略更平直冷淡）；#34 文本自带「ふぁ……」，两版都很有困意，有标版尾音拖曳略更自然。</div>
      <div class="cite-pair">
        <div><div class="meta"><span class="badge m-nocap">无标 #19 冷淡</span></div><audio controls preload="none" src="/samples/full_nocap_1000/19_D_cold.wav"></audio></div>
        <div><div class="meta"><span class="badge m-cap">有标 #19 冷淡</span></div><audio controls preload="none" src="/samples/full_captioned_1000/19_D_cold.wav"></audio></div>
      </div>
    </div>
  </div>
</div>

<script>
const MODEL_META = [
  { name: 'full_nocap_1000', label: '全量 无 Caption (1000步)', cls: 'm-nocap', defaultOn: true },
  { name: 'full_captioned_1000', label: '全量 有 Caption (1000步)', cls: 'm-cap', defaultOn: true },
  { name: 'base_zero_shot', label: 'Base Zero-Shot', cls: 'm-base', defaultOn: false },
];

const AI_BLIND_WINNER = __AI_BLIND_WINNER__;

let groups = [];
let humanVotes = {};
let enabledModels = new Set(MODEL_META.filter(m => m.defaultOn).map(m => m.name));
let activeCat = 'ALL';
let activeGrp = 'ALL';
let activeJudge = 'ALL';
let activeHuman = 'ALL';
let lastKey = '';
let saveStatusText = '自动保存至 outputs/mano_full_ab_noref_eval/human_votes.json';
const noteTimers = new Map();

function esc(s) {
  return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function getAllBaseItems() {
  const byName = new Map(groups.map(g => [g.name, g]));
  const avail = MODEL_META.filter(m => byName.has(m.name)).map(m => ({ ...m, group: byName.get(m.name) }));
  return avail.length ? avail[0].group.items : [];
}

function updateHumanStats() {
  const items = getAllBaseItems();
  const total = items.length || 40;
  let judged = 0, capW = 0, nocapW = 0, tieW = 0, agreed = 0;
  for (const it of items) {
    const v = humanVotes[it.fname];
    if (v && v.winner) {
      judged++;
      if (v.winner === 'full_captioned_1000') capW++;
      else if (v.winner === 'full_nocap_1000') nocapW++;
      else if (v.winner === 'Tie') tieW++;
      if (v.winner === AI_BLIND_WINNER[it.fname]) agreed++;
    }
  }
  const decisive = capW + nocapW;
  const winRate = decisive ? ((capW / decisive) * 100).toFixed(1) + '%' : '-';
  const agreeRate = judged ? ((agreed / judged) * 100).toFixed(1) + '%' : '-';
  document.getElementById('humanStats').innerHTML = `
    <span><b>我的评判进度</b>：<code>${judged} / ${total}</code></span>
    <span><span class="badge m-cap">有标胜 ${capW}</span><span class="badge m-nocap">无标胜 ${nocapW}</span><span class="badge m-base">平局 ${tieW}</span></span>
    <span>去平局有标胜率：<b>${winRate}</b></span>
    <span>与 AI 盲评一致：<b>${agreed}/${judged} (${agreeRate})</b></span>
    <span class="save-hint" id="saveHint">${esc(saveStatusText)}</span>
  `;
}

function matchHumanFilter(fname) {
  const w = (humanVotes[fname] && humanVotes[fname].winner) || '';
  const ai = AI_BLIND_WINNER[fname] || '';
  if (activeHuman === 'ALL') return true;
  if (activeHuman === 'UNJUDGED') return !w;
  if (activeHuman === 'JUDGED') return !!w;
  if (activeHuman === 'AGREE') return !!w && w === ai;
  if (activeHuman === 'DISAGREE') return !!w && w !== ai;
  return w === activeHuman;
}

function renderHumanPills(sourceItems) {
  const cnt = (fn) => sourceItems.filter(it => fn(it.fname)).length;
  const opts = [
    ['ALL', `全部 (${sourceItems.length})`],
    ['UNJUDGED', `未评 (${cnt(f => !(humanVotes[f] && humanVotes[f].winner))})`],
    ['JUDGED', `已评 (${cnt(f => !!(humanVotes[f] && humanVotes[f].winner))})`],
    ['AGREE', `与AI一致 (${cnt(f => { const w = humanVotes[f]?.winner; return !!w && w === AI_BLIND_WINNER[f]; })})`],
    ['DISAGREE', `与AI分歧 (${cnt(f => { const w = humanVotes[f]?.winner; return !!w && w !== AI_BLIND_WINNER[f]; })})`],
    ['full_captioned_1000', `我评：有标胜 (${cnt(f => humanVotes[f]?.winner === 'full_captioned_1000')})`],
    ['full_nocap_1000', `我评：无标胜 (${cnt(f => humanVotes[f]?.winner === 'full_nocap_1000')})`],
    ['Tie', `我评：平局 (${cnt(f => humanVotes[f]?.winner === 'Tie')})`],
  ];
  document.getElementById('humanPills').innerHTML = '<span class="meta">我的评判筛选：</span>' + opts.map(([val, lbl]) => {
    return `<button class="pill ${val===activeHuman?'active':''}" onclick="setHumanFilter('${val}')">${esc(lbl)}</button>`;
  }).join('');
}

function agreeBadgeHtml(fname, winner) {
  if (!winner) return '<span class="save-hint">未标注</span>';
  const ai = AI_BLIND_WINNER[fname] || '';
  if (winner === ai) return '<span class="badge agree">与AI一致</span>';
  return '<span class="badge disagree">与AI分歧</span>';
}

function updateRowVoteUI(fname) {
  const cell = document.querySelector(`td[data-vote-fname="${CSS.escape(fname)}"]`);
  if (!cell) return;
  const v = humanVotes[fname] || { winner: '', note: '' };
  const w = v.winner || '';
  cell.querySelectorAll('.vbtn').forEach(btn => {
    btn.classList.remove('active-cap', 'active-nocap', 'active-tie');
    const val = btn.dataset.val;
    if (val === w) {
      if (val === 'full_captioned_1000') btn.classList.add('active-cap');
      else if (val === 'full_nocap_1000') btn.classList.add('active-nocap');
      else if (val === 'Tie') btn.classList.add('active-tie');
    }
  });
  const metaEl = cell.querySelector('.vote-meta-badge');
  if (metaEl) metaEl.innerHTML = agreeBadgeHtml(fname, w);
}

async function persistSingleVote(fname) {
  const v = humanVotes[fname] || { winner: '', note: '' };
  const hint = document.getElementById('saveHint');
  if (hint) hint.textContent = '保存中...';
  try {
    const res = await fetch('/api/human_votes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fname, winner: v.winner || '', note: v.note || '' }),
    });
    const data = await res.json();
    if (data.ok && data.votes) {
      humanVotes = data.votes;
      saveStatusText = '已保存 ' + new Date().toLocaleTimeString();
      updateHumanStats();
      const baseItems = getAllBaseItems();
      const catFiltered = baseItems.filter(i => activeCat === 'ALL' || i.category === activeCat);
      const grpFiltered = catFiltered.filter(i => activeGrp === 'ALL' || i.group === activeGrp);
      const judgeFiltered = grpFiltered.filter(i => activeJudge === 'ALL' || AI_BLIND_WINNER[i.fname] === activeJudge);
      renderHumanPills(judgeFiltered);
    }
  } catch (e) {
    if (hint) hint.textContent = '保存失败，请检查服务';
  }
}

function setVote(fname, val) {
  const cur = humanVotes[fname] || { winner: '', note: '' };
  const nextWinner = cur.winner === val ? '' : val;
  humanVotes[fname] = { ...cur, winner: nextWinner };
  if (!humanVotes[fname].winner && !humanVotes[fname].note) {
    delete humanVotes[fname];
  }
  updateRowVoteUI(fname);
  updateHumanStats();
  persistSingleVote(fname);
}

function onNoteInput(fname, val) {
  const cur = humanVotes[fname] || { winner: '', note: '' };
  humanVotes[fname] = { ...cur, note: val };
  if (noteTimers.has(fname)) clearTimeout(noteTimers.get(fname));
  noteTimers.set(fname, setTimeout(() => {
    noteTimers.delete(fname);
    persistSingleVote(fname);
  }, 500));
}

function onNoteBlur(fname, val) {
  if (noteTimers.has(fname)) {
    clearTimeout(noteTimers.get(fname));
    noteTimers.delete(fname);
  }
  const cur = humanVotes[fname] || { winner: '', note: '' };
  humanVotes[fname] = { ...cur, note: val };
  persistSingleVote(fname);
}

async function importVotes(inputEl) {
  const file = inputEl.files && inputEl.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    const nextVotes = {};
    if (Array.isArray(parsed.items)) {
      for (const it of parsed.items) {
        if (it.fname && (it.human_winner || it.human_note)) {
          nextVotes[it.fname] = { winner: it.human_winner || '', note: it.human_note || '' };
        }
      }
    } else if (parsed.votes && typeof parsed.votes === 'object') {
      Object.assign(nextVotes, parsed.votes);
    } else if (typeof parsed === 'object' && parsed) {
      Object.assign(nextVotes, parsed);
    }
    const res = await fetch('/api/human_votes', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ votes: nextVotes }),
    });
    const data = await res.json();
    if (data.ok) {
      humanVotes = data.votes || {};
      saveStatusText = '已导入 ' + Object.keys(humanVotes).length + ' 条记录';
      render();
    }
  } catch (e) {
    alert('导入失败：请确认是本页面导出的 JSON 文件。');
  } finally {
    inputEl.value = '';
  }
}

async function clearAllVotes() {
  if (!confirm('确定清空全部 40 题的个人评判和备注吗？建议先点击「导出 JSON」备份。')) return;
  const res = await fetch('/api/human_votes', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ votes: {} }),
  });
  const data = await res.json();
  if (data.ok) {
    humanVotes = {};
    saveStatusText = '已清空全部评判';
    render();
  }
}

function render() {
  const byName = new Map(groups.map(g => [g.name, g]));
  const avail = MODEL_META.filter(m => byName.has(m.name)).map(m => ({ ...m, group: byName.get(m.name) }));
  const togglesEl = document.getElementById('modelToggles');
  if (!avail.length) {
    togglesEl.innerHTML = '等待首个模型生成完毕...';
    return;
  }
  updateHumanStats();
  togglesEl.innerHTML = '<span style="color:var(--muted)">显示模型列：</span>' + avail.map(m => {
    const chk = enabledModels.has(m.name) ? 'checked' : '';
    return `<label><input type="checkbox" ${chk} onchange="toggleModel('${m.name}', this.checked)"><span class="badge ${m.cls}">${esc(m.label)}</span></label>`;
  }).join('');

  const baseItems = avail[0].group.items;
  const cats = ['ALL', ...Array.from(new Set(baseItems.map(i => i.category)))];
  document.getElementById('catPills').innerHTML = cats.map(c => {
    const cnt = c === 'ALL' ? baseItems.length : baseItems.filter(i => i.category === c).length;
    const lbl = c === 'ALL' ? `全部 (${cnt})` : `${c} (${cnt})`;
    return `<button class="pill ${c===activeCat?'active':''}" onclick="setCat('${esc(c)}')">${esc(lbl)}</button>`;
  }).join('');

  const catFiltered = baseItems.filter(i => activeCat === 'ALL' || i.category === activeCat);
  const grps = ['ALL', ...Array.from(new Set(catFiltered.map(i => i.group)))];
  document.getElementById('grpPills').innerHTML = '<span class="meta">子分组：</span>' + grps.map(g => {
    const lbl = g === 'ALL' ? '全部' : g;
    return `<button class="pill ${g===activeGrp?'active':''}" onclick="setGrp('${esc(g)}')">${esc(lbl)}</button>`;
  }).join('');

  const grpFiltered = catFiltered.filter(i => activeGrp === 'ALL' || i.group === activeGrp);
  const cntCap = grpFiltered.filter(i => AI_BLIND_WINNER[i.fname] === 'full_captioned_1000').length;
  const cntNoCap = grpFiltered.filter(i => AI_BLIND_WINNER[i.fname] === 'full_nocap_1000').length;
  const cntTie = grpFiltered.filter(i => AI_BLIND_WINNER[i.fname] === 'Tie').length;
  const judgeOpts = [
    ['ALL', `全部 (${grpFiltered.length})`],
    ['full_captioned_1000', `AI 盲评：有标胜 (${cntCap})`],
    ['full_nocap_1000', `AI 盲评：无标胜 (${cntNoCap})`],
    ['Tie', `AI 盲评：平局 (${cntTie})`],
  ];
  document.getElementById('judgePills').innerHTML = '<span class="meta">AI 盲评筛选：</span>' + judgeOpts.map(([val, lbl]) => {
    return `<button class="pill ${val===activeJudge?'active':''}" onclick="setJudge('${val}')">${esc(lbl)}</button>`;
  }).join('');

  const judgeFiltered = grpFiltered.filter(i => activeJudge === 'ALL' || AI_BLIND_WINNER[i.fname] === activeJudge);
  renderHumanPills(judgeFiltered);

  const cols = avail.filter(m => enabledModels.has(m.name));
  if (!cols.length) {
    document.getElementById('samplesArea').innerHTML = '<div class="meta">请至少勾选 1 个模型列。</div>';
    return;
  }

  const lookup = new Map();
  for (const m of cols) {
    const mp = new Map();
    for (const it of m.group.items) mp.set(it.fname, it);
    lookup.set(m.name, mp);
  }

  const rows = judgeFiltered.filter(i => matchHumanFilter(i.fname));
  let html = '<table><colgroup><col class="prompt-col">' + cols.map(() => '<col class="model-col">').join('') + '<col class="human-col"></colgroup>';
  html += '<thead><tr><th>测试句 &amp; Caption（均无参考音频）</th>' + cols.map(m => `<th><span class="badge ${m.cls}">${esc(m.label)}</span></th>`).join('') + '<th>我的评判（点击即自动保存）</th></tr></thead><tbody>';

  rows.forEach((it) => {
    const origIdx = baseItems.findIndex(x => x.fname === it.fname) + 1;
    const winner = AI_BLIND_WINNER[it.fname] || '';
    const wBadge = winner === 'full_captioned_1000'
      ? '<span class="badge m-cap">AI 盲评：有标胜</span>'
      : winner === 'full_nocap_1000'
      ? '<span class="badge m-nocap">AI 盲评：无标胜</span>'
      : winner === 'Tie'
      ? '<span class="badge m-base">AI 盲评：平局</span>'
      : '';
    html += `<tr><td>
      <div><span class="badge grp">#${String(origIdx).padStart(2,'0')}</span><span class="badge">${esc(it.category)}</span><span class="badge grp">${esc(it.group)}</span>${wBadge}</div>
      <div class="prompt-text">${esc(it.text)}</div>
      <div class="prompt-cap">${esc(it.caption)}<div class="prompt-cap-zh">译：${esc(it.caption_zh)}</div></div>
    </td>`;
    for (const m of cols) {
      const mIt = lookup.get(m.name).get(it.fname);
      let tdCls = '';
      let tag = '';
      if (winner === m.name) {
        tdCls = m.name === 'full_captioned_1000' ? 'win-cap' : 'win-nocap';
        tag = `<div class="win-tag ${m.name === 'full_captioned_1000' ? 'cap' : 'nocap'}">AI 盲评胜出</div>`;
      } else if (winner === 'Tie' && (m.name === 'full_captioned_1000' || m.name === 'full_nocap_1000')) {
        tdCls = 'win-tie';
        tag = '<div class="win-tag tie">AI 盲评平局</div>';
      }
      html += mIt ? `<td class="${tdCls}">${tag}<audio controls preload="none" src="/samples/${esc(mIt.file)}"></audio></td>` : '<td class="meta">生成中...</td>';
    }
    const hv = humanVotes[it.fname] || { winner: '', note: '' };
    const hw = hv.winner || '';
    const fnEsc = esc(it.fname);
    html += `<td data-vote-fname="${fnEsc}">
      <div class="vote-meta"><span class="vote-meta-badge">${agreeBadgeHtml(it.fname, hw)}</span></div>
      <div class="vote-btns">
        <button class="vbtn ${hw==='full_nocap_1000'?'active-nocap':''}" data-val="full_nocap_1000" onclick="setVote('${fnEsc}','full_nocap_1000')">无标胜</button>
        <button class="vbtn ${hw==='Tie'?'active-tie':''}" data-val="Tie" onclick="setVote('${fnEsc}','Tie')">平局</button>
        <button class="vbtn ${hw==='full_captioned_1000'?'active-cap':''}" data-val="full_captioned_1000" onclick="setVote('${fnEsc}','full_captioned_1000')">有标胜</button>
      </div>
      <textarea class="note-input" placeholder="备注（可选，自动保存）" oninput="onNoteInput('${fnEsc}', this.value)" onblur="onNoteBlur('${fnEsc}', this.value)">${esc(hv.note || '')}</textarea>
    </td>`;
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById('samplesArea').innerHTML = html;

  document.querySelectorAll('audio').forEach(a => {
    if (a.dataset.ex) return;
    a.dataset.ex = '1';
    a.addEventListener('play', () => {
      document.querySelectorAll('audio').forEach(o => { if (o !== a && !o.paused) o.pause(); });
    });
  });
}

function toggleModel(n, c) { c ? enabledModels.add(n) : enabledModels.delete(n); render(); }
function setCat(c) { activeCat = c; activeGrp = 'ALL'; render(); }
function setGrp(g) { activeGrp = g; render(); }
function setJudge(j) { activeJudge = j; render(); }
function setHumanFilter(h) { activeHuman = h; render(); }

async function loadVotesOnce() {
  try {
    const res = await fetch('/api/human_votes');
    const data = await res.json();
    humanVotes = data.votes || {};
  } catch (e) {}
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    const b = document.getElementById('evalBadge');
    if (data.evaluating) { b.style.display = 'inline-block'; b.textContent = '正在生成: ' + data.evaluating; }
    else { b.style.display = 'none'; }
    const k = JSON.stringify(data.groups.map(g => `${g.name}:${g.items.length}`));
    if (k !== lastKey) { lastKey = k; groups = data.groups; render(); }
  } catch (e) {}
}
(async () => {
  await loadVotesOnce();
  await refresh();
  setInterval(refresh, 6000);
})();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _send_json(self, obj: dict, status: HTTPStatus = HTTPStatus.OK, filename: str | None = None):
        body = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            page = HTML_PAGE.replace(
                "__AI_BLIND_WINNER__",
                json.dumps(AI_BLIND_WINNER, ensure_ascii=False),
            )
            body = page.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/status":
            loaded = []
            if EVAL_DIR.exists():
                for d in sorted(EVAL_DIR.iterdir()):
                    df = d / "done.json"
                    if df.is_file():
                        loaded.append(json.loads(df.read_text(encoding="utf-8")))
            with _lock:
                ev = _currently_evaluating
            self._send_json({"evaluating": ev, "groups": loaded})
            return
        if path == "/api/human_votes":
            self._send_json({"votes": load_human_votes(), "saved_path": str(HUMAN_VOTES_FILE)})
            return
        if path == "/api/export":
            qs = parse_qs(parsed.query)
            fmt = (qs.get("format") or ["json"])[0].lower()
            payload = build_export_payload(load_human_votes())
            if fmt == "csv":
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "index", "fname", "category", "group", "text", "caption", "caption_zh",
                    "ai_blind_winner", "human_winner", "agree_with_ai", "human_note", "updated_at",
                ])
                for it in payload["items"]:
                    writer.writerow([
                        it["index"], it["fname"], it["category"], it["group"],
                        it["text"], it["caption"], it["caption_zh"],
                        it["ai_blind_winner"], it["human_winner"],
                        "" if it["agree_with_ai"] is None else ("1" if it["agree_with_ai"] else "0"),
                        it["human_note"], it["updated_at"],
                    ])
                body = ("\ufeff" + buf.getvalue()).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", 'attachment; filename="human_eval_noref_40.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._send_json(payload, filename="human_eval_noref_40.json")
            return
        if path.startswith("/samples/"):
            rel = unquote(path[len("/samples/"):]).lstrip("/")
            fp = (EVAL_DIR / rel).resolve()
            if not str(fp).startswith(str(EVAL_DIR.resolve())) or not fp.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = fp.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/human_votes":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json({"ok": False, "error": "invalid json"}, status=HTTPStatus.BAD_REQUEST)
            return

        now_iso = datetime.now(timezone.utc).isoformat()
        if isinstance(body.get("votes"), dict):
            cleaned: dict[str, dict[str, str]] = {}
            for fn, val in body["votes"].items():
                if fn not in PROMPT_FNAMES or not isinstance(val, dict):
                    continue
                w = str(val.get("winner", "") or "")
                n = str(val.get("note", "") or "")
                if w not in VALID_WINNERS:
                    continue
                if w or n.strip():
                    cleaned[fn] = {"winner": w, "note": n, "updated_at": str(val.get("updated_at") or now_iso)}
            save_human_votes(cleaned)
            self._send_json({"ok": True, "votes": cleaned, "saved_path": str(HUMAN_VOTES_FILE)})
            return

        fn = str(body.get("fname", "") or "")
        w = str(body.get("winner", "") or "")
        n = str(body.get("note", "") or "")
        if fn not in PROMPT_FNAMES or w not in VALID_WINNERS:
            self._send_json({"ok": False, "error": "invalid fname or winner"}, status=HTTPStatus.BAD_REQUEST)
            return
        votes = load_human_votes()
        if w or n.strip():
            votes[fn] = {"winner": w, "note": n, "updated_at": now_iso}
        else:
            votes.pop(fn, None)
        save_human_votes(votes)
        self._send_json({"ok": True, "votes": votes, "saved_path": str(HUMAN_VOTES_FILE)})


def main():
    t = threading.Thread(target=worker_loop, daemon=True)
    t.start()
    server = ThreadingHTTPServer(("0.0.0.0", 8765), Handler)
    print("No-ref 40-item benchmark server running at http://0.0.0.0:8765/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
