# Shiny Colors 29 角色 Irodori-TTS 全量微调与标注计划

> 本文档作为主仓库的可追踪总控计划与执行日志，持续更新进度。

## 一、核心设计决策

- **角色范围**：28 位偶像 + はづき（共 **29 角色**），时长放宽至 **`2–15s`**，按 `speaker_id + 规范化文本` 去重。
- **均衡度约束（`min : max >= 1/3`）**：
  - 单角色上限 **`6,480`**（下采样优先级：`已标有caption > 高情绪 > 平静日常`）；
  - はづき（`769` 条）采用 **`3×` 过采样**（`2,307` 行），与羽那（`2,163`）/ ルカ（`2,171`）持平；
  - 最终有效最小/最大比 `2,163 : 6,480 = 0.334 > 1/3`。
- **标注体系与仓库边界**：
  - `voice-caption-annotator/` 保持 100% 通用开源库，零 SC 耦合；所有 SC 工作区与主库放在主仓库 `datasets/` 下。
  - **Wave A (`characters/` P0+P1，28 位偶像，7,669 条)**：走三段式流程 `Pass 1 (LLM low 听写+标) -> Whisper large-v3 (单次 GPU) -> Pass 2 (LLM medium 双源原文仲裁+复核)`。
  - **Wave B (`events/` 剧情分层 7:3，29 角色，~9,200 条)**：走二段式流程 `Pass 1 (LLM low) -> Pass 2 (LLM medium)`，70% 高情绪核心回 + 30% 平静/明朗日常回。
  - **并发能力**：已实测 `agy` **10/10 并发无报错**，批量标注采用 `--workers 10`。
  - **并行加速与双版本消融（`v1_lite` vs `v2_full`）**：
    - Wave B 跑批期间利用空闲 RTX 4090 提前准备并预跑 Wave A-P1+P2 的本地 Whisper large-v3；完成后用 `scripts/whisper_cache_ctl.py` 暂存到 `.whisper_cache_p1.json` 并清空 JSON 中的 `whisper_text`，保障 Pass 1 盲听独立性与 LLM 阶段 0% GPU 占用，Pass 1 完成后瞬时回填再跑 Pass 2。
    - Wave A-P1+P2 LLM API 标注（0% GPU）与 GPU 训练彻底解耦：**`v1_lite`**（`Wave A-P0 + Wave B`，10,000 步，`configs/dataset_sc29_v1_lite.yaml` → `outputs/sc29_v1_lite/`）先于 4090 并行开训；待 Wave A-P1+P2 入库后仅需增量提取 ~1.4 万条新增 latents，再训 **`v2_full`**（`Wave A-P0+P1+P2 + Wave B`，10,000 步，`configs/dataset_sc29_v1.yaml` → `outputs/sc29_v1_full/`），最终执行 **Base vs `v1_lite` vs `v2_full`** 统一三方评测。
- **数据管理底座**：
  - 主库：`datasets/catalog.sqlite`（由 `scripts/sync_catalog.py` 幂等同步）
  - 共享特征池：`datasets/shared_latents/`（由 `scripts/extract_shared_latents.py` 直接从原 `.m4a` 流式 `ffmpeg` 裁剪并提取 DACVAE `.pt`，跳过 ~75 GB 中间 WAV 落盘；Phase 3 每波入库后由 `scripts/prune_synced_wavs.py` 清理已同步中间 WAV 以节约空间）
  - 配方构建器：`scripts/build_dataset.py --recipe configs/dataset_sc29_v1.yaml` → 输出 `datasets/sc29_v1/{train,valid}_manifest.jsonl`

---

## 二、阶段任务清单 (Checklist)

### Phase 0：前期验证与基建
- [x] 真乃 12.6% 分层标注（31 话 / 710 条）与全量微调 A/B 验证（双盲 69.4% 胜率，`p = 0.0288`）
- [x] MongoDB `shinycolors.voices` 增量 upsert（+19,085 条，总计 254,752 条）并补齐全部 `duration`
- [x] 搭建三段式 `characters/` ASR+Caption 流水线并完成真乃 `casting_voice` 8 条端到端试跑（0 错 0 警告）
- [x] 清理 `voice-caption-annotator` 硬编码与 SC 专有词，完成双仓库提交（`3bac17d` / `a0bd05b`）
- [x] 实测 `agy` **10 并发**压力测试（`10/10` 成功，总耗时 `11.7s`）

### Phase 1：SQLite 主库与构建器底座
- [x] 编写 `scripts/sync_catalog.py`、`scripts/build_dataset.py`、`configs/dataset_sc29_v1.yaml`
- [x] 迁移已有标注工作区至 `datasets/annotator_work/{events,characters}/`
- [x] 首次灌入 `datasets/catalog.sqlite`（`174,299` 条剧情 + `8,052` 条角色语音 + `746` 条已复核标注）
- [x] 提交 Phase 1 脚本与本计划文档至主仓库

### Phase 2：28 角色 7:3 分层剧情并发选话（真乃已完成）
- [x] Subagent Batch 1（`hiori`, `meguru`, `kogane`, `mamimi`, `sakuya`, `yuika`, `kiriko`）完成选话
- [x] Subagent Batch 2（`kaho`, `chiyoko`, `juri`, `rinze`, `natsuha`）完成选话
- [x] Subagent Batch 3（`amana`, `tenka`, `chiyuki`, `asahi`, `fuyuko`, `mei`）完成选话
- [x] Subagent Batch 4（`toru`, `madoka`, `koito`, `hinana`, `nichika`, `mikoto`）完成选话
- [x] Subagent Batch 5（`luca`, `hana`, `haruki`, `hazuki`）完成选话
- [x] 汇总校验并落盘 `datasets/selection_plan_sc29.json`（28 角色 / 385 话 / 9,026 条，+ 真乃 710 = **9,736 条**；后修正 `luca 302600302 -> 102600302` 合计 **9,759 条**）

### Phase 3：分波次 10 并发批量标注
- [x] **Wave A-0**：导出全部 28 位偶像 `characters/` P0+P1 批次包（`490` 批 / `7,669` 条），并完成 P0 全部 `208` 批预处理（0 失败）
- [x] **Wave A-P0**：完成 P0 跨人设 `casting_voice` + 跨角色多人对话 `talking_multi` + 显式情绪词（**`209/209` 批全部完成**，`3,387` reviewed + `1` rejected，`0` 失败，已同步 SQLite 并清理 `3.59 GiB` 中间 WAV）
- [x] **Wave A-P1+P2 预处理与 Whisper 预跑**：完成 `289` 批预处理与 Batched GPU Whisper 全量重洗（`4,412/4,412` 条 `100%` 非空、`0.134s/条` 总计 `593.3s`、`0` 重复幻觉，提速 ~11×），并将碎片包合批重构为 `249` 批（`≤20 条 / ≤120s`，写入 `datasets/annotator_work/characters_p1/`、清理旧 `prepared` 目录、冬优子 `10/10` 批注规则、暂存 `.whisper_cache_p1.json` 保障 Pass 1 盲听）
- [ ] **Wave A-P1+P2 LLM 标注入库**：完成 P1+P2 共 `249` 批 / `4,412` 条三段式标注（**`5` 批生产小验证已全数通过**，余 `244` 批 10 并发 Pass 1 运行中 → 瞬时回填 Whisper → Pass 2 双源仲裁），回写 SQLite 后清理已同步 WAV（与 GPU `v1_lite` 并行）
- [x] **Wave B-0**：按 `selection_plan_sc29.json` 导出并预处理其余 28 角色剧情批次包（`374/374` 包完成，`0` 失败，`400100103__hiori` 独立隔离，11 批含冬优子已注入规则）
- [x] **Wave B-1～B-4**：10 并发全量跑完 28 角色 7:3 分层剧情 + 冬优子 `101800103` 按方案 A 重标（**`406/406` 话全部 `reviewed`**，`0` 错 `0` 警告；`3` 话游戏回忆重复 ID 已清理，冬优子 `12/12` 话 `336/336` 条 **100%** 命中方案 A）
- [x] **Wave B 入库**：运行 `sync_catalog.py` 全部回写 `catalog.sqlite`（`event reviewed = 9,756`，`character reviewed = 3,387`），并清理已同步剧情 WAV（释放 `8.14 GiB`，当前 `614 GiB` 可用）

### Phase 4：全团共享 DACVAE Latent 提取与双版本清单构建（`v1_lite` → `v2_full` 增量复用）
- [ ] **`v1_lite` 底座构建**：Wave B 入库后运行 `build_dataset.py --recipe configs/dataset_sc29_v1_lite.yaml` 锁定底座清单（已完成 `--plan-only` 验证：`29` 角色 / `142,383` 唯一条 / `166,982` 训练行，`min/max = 1,923/5,767 = 0.3334+`），RTX 4090 直提共享 `shared_latents/` 并校验 `LatentTextDataset`
- [ ] **`v2_full` 增量构建**：Wave A-P1+P2 入库后运行 `build_dataset.py --recipe configs/dataset_sc29_v1.yaml`（`min:max >= 1/3` 抬升上限），仅需 **~6 分钟增量补提 ~1.4 万条** 新增 `shared_latents/` 并生成最终 `datasets/sc29_v1/{train,valid}_manifest.jsonl`

### Phase 5：29 角色双版本全量微调（并行 `v1_lite` + `v2_full`）与三方评测
- [ ] **`v1_lite` 并行微调**：在 Wave A-P1+P2 API 标注同时启动 `outputs/sc29_v1_lite/` 全量微调（10,000 步 Muon, `bs=8, grad_accum=4, grad_ckpt=true`, `checkpoint_best_n=1`），导出 `v1_lite/model_best.safetensors` 并清理大 `.pt`
- [ ] **`v2_full` 完整微调**：增量 latent 就绪后启动 `outputs/sc29_v1_full/` 完整版 10,000 步微调，导出 `v2_full/model_best.safetensors` 并清理大 `.pt`
- [ ] **三方统一验收**：搭建多角色 + 冬优子双人格对比评估台，同批对比 **Base vs `v1_lite` vs `v2_full`**（ASR CER + 情绪/人格切换控制力 + 人工试听）

---

## 三、29 角色标注进度追踪表

| # | 角色 | ID | `characters` P0+P1 | `events` 7:3 选话 | `events` 标注 | SQLite 同步 |
|---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | 真乃 | `mano` | P0 已完 (`casting`+`multi`) / P1 Whisper 预跑中 | 已完成 (31话) | **709/709** | P0+剧情已同步 |
| 2 | 灯織 | `hiori` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **333/333** | P0+剧情已同步 |
| 3 | めぐる | `meguru` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **325/325** | P0+剧情已同步 |
| 4 | 恋鐘 | `kogane` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **341/341** | P0+剧情已同步 |
| 5 | 摩美々 | `mamimi` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **328/328** | P0+剧情已同步 |
| 6 | 咲耶 | `sakuya` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **332/332** | P0+剧情已同步 |
| 7 | 結華 | `yuika` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **328/328** | P0+剧情已同步 |
| 8 | 霧子 | `kiriko` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **331/331** | P0+剧情已同步 |
| 9 | 果穂 | `kaho` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **320/320** | P0+剧情已同步 |
| 10 | 智代子 | `chiyoko` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **326/326** | P0+剧情已同步 |
| 11 | 樹里 | `juri` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **329/329** | P0+剧情已同步 |
| 12 | 凛世 | `rinze` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **349/349** | P0+剧情已同步 |
| 13 | 夏葉 | `natsuha` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **303/303** | P0+剧情已同步 |
| 14 | 甘奈 | `amana` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **344/344** | P0+剧情已同步 |
| 15 | 甜花 | `tenka` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **328/328** | P0+剧情已同步 |
| 16 | 千雪 | `chiyuki` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **337/337** | P0+剧情已同步 |
| 17 | あさひ | `asahi` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **328/328** | P0+剧情已同步 |
| 18 | 冬優子 | `fuyuko` | P0 已完(129条触发词) / P1 Whisper 预跑中 | 已完成 (12话) | **336/336 (100% 方案A)** | P0+剧情已同步 |
| 19 | 愛依 | `mei` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **340/340** | P0+剧情已同步 |
| 20 | 透 | `toru` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **344/344** | P0+剧情已同步 |
| 21 | 円香 | `madoka` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **331/331** | P0+剧情已同步 |
| 22 | 小糸 | `koito` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **335/335** | P0+剧情已同步 |
| 23 | 雛菜 | `hinana` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **348/348** | P0+剧情已同步 |
| 24 | にちか | `nichika` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **343/343** | P0+剧情已同步 |
| 25 | 美琴 | `mikoto` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **324/324** | P0+剧情已同步 |
| 26 | ルカ | `luca` | P0 已完 / P1+P2 Whisper 预跑中 | 已完成 | **306/306** | P0+剧情已同步 |
| 27 | 羽那 | `hana` | P0 已完 / P1+P2 Whisper 预跑中 | 已完成 | **314/314** | P0+剧情已同步 |
| 28 | はるき | `haruki` | P0 已完 / P1 Whisper 预跑中 | 已完成 | **329/329** | P0+剧情已同步 |
| 29 | はづき | `hazuki` | 无 P0/P1 (3x 过采样) | 已完成 | **115/115** | 剧情已同步 |

---

## 四、执行日志 (Execution Log)

- **2026-11-21 21:56** — 完成 MongoDB `shinycolors.voices` 增量 upsert（`+19,085` 条，总库 `254,752` 条）及全部 `duration` 补算。
- **2026-11-21 23:02** — 真乃 `char_001_mano_casting`（8 条跨人设反差语音）三段式端到端试跑通过（0 错 0 警告；#07 触发 Pass 1 / Whisper 双源分歧并由 Pass 2 单听精准融合修正）。
- **2026-11-21 23:19** — `voice-caption-annotator`（`3bac17d`）与主仓库（`a0bd05b`）完成干净提交；`agy` 10 并发压力测试 `10/10` 全部成功（总耗时 `11.7s`）。
- **2026-11-21 23:22** — Phase 1 SQLite 主库 `datasets/catalog.sqlite` 初始化完成（`174,299` 剧情 + `8,052` 角色语音 + `746` 已复核标注）；Phase 2 5 路并行子代理选话与 Wave A `characters/` P0+P1 批次导出已同步启动。
- **2026-11-21 23:25** — Wave A `characters/` P0+P1 全部导出完成（`490` 批次 / `7,669` 条）；创建总控计划文档 `docs/SC29_TRAINING_PLAN.md` 并提交 Phase 1。
- **2026-11-21 23:26** — Wave A-P0 全部 `208/208` 批次预处理完成（0 失败），已以 `--workers 10` 启动三段式标注（`task-1455`）。
- **2026-11-21 23:28** — Phase 2 全部 5 路子代理选话完成并经 SQLite 校验去重，落盘 `datasets/selection_plan_sc29.json`（其余 28 角色共 `385` 话 / `9,026` 条，高情绪占比严格控制在 `68%–78%`，加上真乃共 `9,736` 条剧情标注）。
- **2026-11-21 23:39** — 冬优子双面人格触发词（方案 A：`猫かぶりな営業口調で、` / `素の本音口調で、` / `営業口調から素が漏れ出しながら、` + `persona:*`）已写入 `configs/speaker_caption_rules/fuyuko.md` 并注入全部 `018_fuyuko` 批次。Wave A-P0 首轮巡检：10 并发无报错，已完成 `13/208` 批 Pass 1（0 invalid，单批 `150s–303s`），3 批撞 `600s` 超时待末尾补跑。
- **2026-11-22 00:05** — **Wave B-0 完成 & 磁盘瘦身**：修复 `selection_plan_sc29.json` 中 `luca` 缺失编号 `302600302 -> 102600302`（23 条日常冷静回，修正后全团计划剧情总计 `9,759` 条），`prepare_wave_b.py` 并发完成全部 `374/374` 新剧情包导出与预处理（`0` 失败，`400100103__hiori` 隔离保护真乃已复核条目，11 个含冬优子的剧情包均已注入方案 A 规则）。清理 `mano_full_*` 冗余 `.pt` 检查点与陈旧 `voice-caption-annotator/work` WAV，建立 Phase 3 分波次入库后清理已同步中间 WAV 与 Phase 4 直接管道提取 `.pt` 的磁盘节约策略。
- **2026-11-22 07:38** — **Wave A-P0 全部收官 & Wave B 启动**：`task-1733` 将遗留 8 个超时批全部补跑通过（总计 `209/209` `reviewed`，`0` 失败），入库 `catalog.sqlite` 后 `character reviewed = 3,387`（冬优子 `145` 条中 `129` 条严格命中方案 A 触发词，余为戏中戏 `casting` / 幼年变声角色台词），并安全清理全部 P0 中间 WAV（累计省 `3.59 GiB`，当前 `610 GiB Avail`）。重置冬优子 `101800103` 按方案 A 重跑，已于 10 并发启动全部 `375` 话 Wave B 二段式标注。
- **2026-11-22 11:10** — **启用并行 GPU/API 双版本路线（`v1_lite` + `v2_full`）**：更新总控脚本 `scripts/sc29_runner.py` 与 `scripts/whisper_cache_ctl.py`，趁 Wave B（已过 `76%`）期间 4090 空闲立即启动 Wave A-P1+P2 `289` 批预处理与本地 Whisper 预跑，并暂存至 `.whisper_cache_p1.json` 保障 Pass 1 盲听独立；后续 Wave B 入库后在 4090 并行开训 `v1_lite`（10,000 步）同时跑 Wave A-P1+P2 API 标注，最终增量提特征完成 `v2_full` 并三方评测（Base vs `v1_lite` vs `v2_full`）。
- **2026-11-22 12:28** — **Wave B 全量收官 & 优化 Pilot 落地**：Wave B 406/406 话全部 `reviewed`（清理 3 话共 4 条原作回忆重复 `id` 并 7 并发补完 Pass 2），`sync_catalog.py` 同步后 `event reviewed = 9,756`、`character reviewed = 3,387`，**冬优子全部 12/12 话（336/336 条）100% 命中方案 A 前缀与 `persona:*` 标签**；安全清理 375 话中间 WAV 释放 **`8.14 GiB`**（磁盘 `614 GiB` 可用）。`v1_lite` `--plan-only` 验证通过：`29` 角色 / `142,383` 唯一条 / `166,982` 训练行（`min/max = 1,923/5,767 = 0.3334+`，`cap_ratio=23.3%`）。完成 `pilot_p1_opt` 3 批对照验证：Pass 2 单条耗时 `-37.8%`（`9.20s → 5.72s/条`）、单听率 `-40%`（`59.5% → 36.0%`）、`context.md` `-89%`（`24.8KB → 2.1–3.1KB`）、`patch_annotations.py` 局部回写 `0` 错 `0` 警告；已将 `run.py` 断尾 401 自愈、`reviewer` 风险单听收紧、29 人精简声演速查卡与 `≤20 条 / ≤120s` 合批优化正式合入双仓库。
- **2026-11-22 13:02** — **Batched Whisper 11× 提速 & 5 批生产小验证过关，GPU+API 双线全面开跑**：`whisper_transcribe.py` 升级为 GPU Batch=16 并消除多窗重复幻觉，在 **`593.3s`（`0.134s/条`，0 fallback）** 全洗 `289` 批 / `4,412` 条实现 **100.0% 非空、`0` 压缩比异常**；合批重构为 `249` 批写入 `datasets/annotator_work/characters_p1/` 并加固 `whisper_cache_ctl.py` 仅回填 `pass1_done` 批。`5` 批生产小验证（`fuyuko`、`hana P2`、`luca P2`、`kogane`、`nichika`，97 条）全数通过（Pass 1 均 **`99.8s`**、Pass 2 均 **`62.7s`**，整体 **`8.4s/条` = 3.0× 提速**、单听率 `39.2%`、`consensus 66.0%`、冬优子 `17/17` **100%** 达标且仅 `35.3%` 单听、自动捉出并修正 `hana` 10 条漂移）。现已同步启动 **RTX 4090 提取 `v1_lite` 142,351 条 latents（`~39.3 utt/s`，ETA ~54m）** 与 **Wave A-P1+P2 10 并发 Pass 1（余 244 批）**。
