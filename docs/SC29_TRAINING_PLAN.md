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
- **数据管理底座**：
  - 主库：`datasets/catalog.sqlite`（由 `scripts/sync_catalog.py` 幂等同步）
  - 共享特征池：`datasets/shared_wavs/` + `datasets/shared_latents/`（每条语音终身只转码/提特征一次）
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
- [x] 汇总校验并落盘 `datasets/selection_plan_sc29.json`（28 角色 / 385 话 / 9,026 条，+ 真乃 710 = **9,736 条**）

### Phase 3：分波次 10 并发批量标注
- [x] **Wave A-0**：导出全部 28 位偶像 `characters/` P0+P1 批次包（`490` 批 / `7,669` 条），并完成 P0 全部 `208` 批预处理（0 失败）
- [ ] **Wave A-P0**：完成 P0 跨人设 `casting_voice` + 跨角色多人对话 `talking_multi` + 显式情绪词（`209` 批 / `3,388` 条，三段式，**10 并发运行中**）
- [ ] **Wave A-P1**：完成 P1 小游戏夸张情绪 + 互动反应 + Live/试镜 + 信赖度 + 评级梯度（`4,281` 条，三段式）
- [ ] **Wave A 入库**：运行 `sync_catalog.py` 回写 SQLite 并确认 `min:max >= 1/3`
- [ ] **Wave B-0**：按 `selection_plan_sc29.json` 导出并预处理其余 28 角色剧情批次包
- [ ] **Wave B-1～B-4**：分组合 10 并发跑完 28 角色 7:3 分层剧情（`~8,500` 条，Pass 1 + Pass 2）
- [ ] **Wave B 入库**：运行 `sync_catalog.py` 全部回写 SQLite，核验 0 错率与总有标占比

### Phase 4：全团共享音频转码与 DACVAE Latent 提取
- [ ] 运行 `build_dataset.py --plan-only` 锁定 `selected_voice_keys.txt`（~15.4 万条）
- [ ] 批量 `ffmpeg` 去首尾静音转 48kHz PCM WAV → `datasets/shared_wavs/`
- [ ] RTX 4090 批量提取 `Semantic-DACVAE-Japanese-32dim` latents → `datasets/shared_latents/` 并回写 SQLite
- [ ] 运行 `build_dataset.py` 生成最终 `datasets/sc29_v1/{train,valid}_manifest.jsonl` 与 `stats.json`

### Phase 5：29 角色全量微调与多角色验证
- [ ] 配置 29 角色全量微调 YAML（Muon, `bs=8, grad_accum=4, grad_ckpt=true`）并启动训练
- [ ] 搭建多角色 `--no-ref` 情绪/反差/音色分离度抽测对比台并验收

---

## 三、29 角色标注进度追踪表

| # | 角色 | ID | `characters` P0+P1 | `events` 7:3 选话 | `events` 标注 | SQLite 同步 |
|---:|:---|:---|:---:|:---:|:---:|:---:|
| 1 | 真乃 | `mano` | `casting` 8/8 (试跑完成) | 已完成 (31话) | **710/710** | 已同步 |
| 2 | 灯織 | `hiori` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 3 | めぐる | `meguru` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 4 | 恋鐘 | `kogane` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 5 | 摩美々 | `mamimi` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 6 | 咲耶 | `sakuya` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 7 | 結華 | `yuika` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 8 | 霧子 | `kiriko` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 9 | 果穂 | `kaho` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 10 | 智代子 | `chiyoko` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 11 | 樹里 | `juri` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 12 | 凛世 | `rinze` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 13 | 夏葉 | `natsuha` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 14 | 甘奈 | `amana` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 15 | 甜花 | `tenka` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 16 | 千雪 | `chiyuki` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 17 | あさひ | `asahi` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 18 | 冬優子 | `fuyuko` | 待跑 | 选话中 | 29/29 (`101800103`) | 已同步 |
| 19 | 愛依 | `mei` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 20 | 透 | `toru` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 21 | 円香 | `madoka` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 22 | 小糸 | `koito` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 23 | 雛菜 | `hinana` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 24 | にちか | `nichika` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 25 | 美琴 | `mikoto` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 26 | ルカ | `luca` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 27 | 羽那 | `hana` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 28 | はるき | `haruki` | 待跑 | 选话中 | 待跑 | 基线已入库 |
| 29 | はづき | `hazuki` | 无 P0/P1 (3x 过采样) | 选话中 | 待跑 | 基线已入库 |

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
