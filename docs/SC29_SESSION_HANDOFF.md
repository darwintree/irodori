# SC29 项目现状与新会话交接文档

- **快照时间**：`2026-11-22 22:57 CST`
- **工作区路径**：`/home/darwinwsl/code/irodori`
- **Git 基线**：分支 `main @ 4b5a26c`（含 `scripts/sc29_runner.py` 未提交本地加固改动及本文档）

---

## 1. 项目目标与当前状态总览

本项目面向《偶像大师 Shiny Colors》**29 角色**（28 位偶像 `001`–`028` + 七草羽月 `901`），基于 `Irodori-TTS-v4.1-Small` 构建三段式路线：
1. **29 角色全量微调 + Caption 风格/双面人格控制**（重点验证黛冬优子 表/里 双人格及情绪/语气可控）；
2. **单角色 49 KB Speaker Inversion（`--ref-embed`）**，消除裸 `--no-ref` 串角色问题，并在推理时完全跳过参考音频 DACVAE 编码与 8 层 `speaker_encoder`；
3. **未来纯 CPU 实时服务化（MeanFlow 4-step 蒸馏 + `--ref-embed`）**，实现近零成本角色切换的本地 CPU TTS 推理。

| 阶段 | 内容摘要 | 当前状态 |
| --- | --- | --- |
| **Phases 0–4：数据、标注与特征** | 精标入库 `17,555` 条（`character=7,799` + `event=9,756`；冬优子方案 A `649/665 = 97.6%`）；共享 DACVAE latents 写入 `datasets/shared_latents/`；正式清单含 `29` 角色 / `154,242` 唯一片段 / `17,406` 精标 / train `187,216` 行 / valid `3,092` 行（`cap_ratio=27.7%`, `min/max=0.3334`） | **已完成** |
| **Phase 5：`v1_lite` / `v2_full` 微调与评测** | `v1_lite@2000` 与 `v2_full@10000`（WSD 调度：`warmup=500, stable=7500, decay=8001→10000`）训练完成；4 路离线 Whisper large-v3 客观评测 + 人工试听完成 | **已完成** |
| **Phase 6a：`s10000` 试点与 CPU 压测** | 基于 `step=10000` 完成 `fuyuko` / `mano` / `luca` 3 角色 Speaker Inversion 试点、`--ref` vs `--ref-embed` vs `--no-ref` 三模式评测、`i5-13600K` 纯 CPU FP32 延迟压测及 4090/H100 MeanFlow 配置预置 | **已完成** |
| **工作区整理** | 导出 `model_s8000.safetensors`；清理 2 份重复 `step=10000` 大体积 `.pt` 释放 `13.8 GiB`；保留 `mano_*` 实验与 `5k/8k/10k` 单拷贝 `.pt`；停止全部后台预览 Web 服务 | **已完成** |
| **Phase 6b：`s8000` 全量 29 角色 Embedding** | 基于 `outputs/sc29_v1_full/model_s8000.safetensors` 训练全部 29 角色 `--ref-embed` | **进行中（断点：3 / 29 完成，第 4 个 `chiyuki` 两次因服务重启中断，余 26 个，预计 ~3.1 小时）** |
| **Phase 7：MeanFlow 4-step 蒸馏** | 4090 / H100 蒸馏训练配置已预制（`configs/train_sc29_meanflow_{4090,h100}.yaml`） | **用户明确暂缓（待后续指令再启动）** |

---

## 2. 核心实验结论与已定决策

### 2.1 为什么拍板选用 `v2_full@8000`（`model_s8000.safetensors`）作为底座

- `train_sc29_v1_full.yaml` 使用 WSD 调度（`warmup_steps=500`, `stable_steps=7500`, `max_steps=10000`），余弦退火冷却发生在 `8001 → 10000` 步：
  - 验证集 `val_loss` 曲线：`1k: 0.766201` → `5k: 0.749625` → `6k: 0.749065` → `7k: 0.748776` → **`8k: 0.745783`** → `9k: 0.745967` → **`10k: 0.744451`**（`dur_mae=10.44`）。
- 4 路离线 Whisper large-v3 客观评测（8 角色联合零泄漏参考集 × 33 句 = 132 条 WAV，详见 `outputs/sc29_v1_eval/summary.json`）：

| 模型 | Overall CER | Sec A CER（冬优子双面） | Sec B CER（多角色泛化） | 円香 (`madoka`) CER | 冬优子 F0（表 / 里 / Δ Hz） |
| --- | :---: | :---: | :---: | :---: | :---: |
| `base`（原版 `v4.1-Small`） | `8.03%` | `7.95%` | `8.06%` | `1.59%` | `283.1 / 238.2 / Δ 44.9 Hz` |
| `v1_lite_s2000` | `7.75%` | `7.02%` | `8.02%` | `0.00%` | `392.0 / 257.4 / Δ 134.6 Hz` |
| **`v2_full_s8000`（已选定底座）** | **`8.09%`** | **`7.45%`** | **`8.33%`** | **`0.00%`** | **`398.5 / 272.6 / Δ 125.9 Hz`** |
| `v2_full_s10000` | `8.96%` | `8.87%` | `9.00%` | `4.76%` | `449.5 / 337.1 / Δ 112.4 Hz` |

- **结论**：`v2_full_s10000` 出现轻微字错率反弹及冬优子整体音高上漂过拟合倾向；用户盲听/实听 `8000` 与 `10000` 无明显听感差别，因此最终选定更稳健的 **`step=8000`（`model_s8000.safetensors`）** 作为全量 29 角色 Speaker Inversion 底座。

### 2.2 为什么必须使用 49 KB `--ref-embed` 而不能使用裸 `--no-ref`

基于 `s10000` 与 3 角色（`fuyuko` / `mano` / `luca`）18 句 × 3 模式 = 54 条评测（`outputs/sc29_v1_eval_embed/summary.json`）：

| 推理模式 | Overall CER | Sec A CER | Sec B CER | `luca` CER | 冬优子 F0（表 / 里 / Δ Hz） |
| --- | :---: | :---: | :---: | :---: | :---: |
| `v2_full_ref`（传统参考音频） | `7.18%` | `8.87%` | `5.50%` | `9.26%` | `449.5 / 337.1 / Δ 112.4 Hz` |
| **`v2_full_embed`（49 KB `--ref-embed`）** | **`6.57%`** | `8.87%` | **`4.26%`** | `5.56%` | `432.2 / 323.9 / Δ 108.3 Hz` |
| `v2_full_noref`（裸 `--no-ref`） | `6.79%` | `7.95%` | `5.64%` | `3.70%` | `405.7 / 299.5 / Δ 106.2 Hz` |

- **串角色实证**：用户实听及客观指标共同证实，裸 `--no-ref` 在**无明显角色口癖的非典型句**上会严重串角色——典型案例如 **`B_luca_bright_joy`**（ルカ柔和感谢台词）：
  - `v2_full_ref`：F0 中位数 `171.1 Hz`
  - `v2_full_embed`（49 KB `--ref-embed`）：F0 中位数 **`162.9 Hz`**（CER `0.0%`），稳稳锁住 ルカ 低音区声线；
  - `v2_full_noref`（裸 `--no-ref`）：F0 中位数飙升至 **`286.0 Hz`**（**`+123.1 Hz`** 漂移成泛用元气女声）。
- **结论**：每个角色只需 **`49,264 B`（49 KB）** `.speaker.safetensors` 即可取代参考音频；推理时既锁死声线、降低 CER，又完全绕过 DACVAE 参考编码与 8 层 `speaker_encoder`。

### 2.3 CPU-only 延迟压测结论

- 测试环境：本机 **`i5-13600K`**、10 线程、纯 CPU `FP32`（未量化）、输入句长 `4.96s`、搭配 49 KB `--ref-embed`（运行自 `python3 scripts/sc29_runner.py bench-cpu`）：
  - **MeanFlow 4-step 等效（4 NFE）**：总耗时 **`3.86s`**（**`RTF = 0.78×`**，未量化已快于实时；其中 `prepare_reference=0.2ms`, `predict_duration=770.5ms`, `sample_rf=1544.9ms`, `decode_latent=1539.6ms`）；
  - **Sway 8-step（`joint` CFG）**：总耗时 **`5.67s`**（`RTF = 1.14×`）；
  - **标准 RF 40-step（`independent` CFG）**：总耗时 **`17.97s`**（`RTF = 3.62×`）。

---

## 3. 关键资产与文件路径索引

- **模型权重（`/home/darwinwsl/code/irodori/outputs/`）**
  - `/home/darwinwsl/code/irodori/outputs/sc29_v1_full/model_s8000.safetensors`（**当前主力底座**，`2.85 GiB` / `ls -lh` 显示 `2.9G`，714 张量）
  - `/home/darwinwsl/code/irodori/outputs/sc29_v1_full/model_best.safetensors`（`step=10000`，`2.85 GiB`）
  - `/home/darwinwsl/code/irodori/outputs/sc29_v1_lite/model_best.safetensors`（`step=2000`，`2.85 GiB`）
  - 保留单拷贝 `.pt` 检查点（可断点续训/回退）：
    - `/home/darwinwsl/code/irodori/outputs/sc29_v1_full/kept/checkpoint_0005000.pt`（`6.9G`）
    - `/home/darwinwsl/code/irodori/outputs/sc29_v1_full/kept/checkpoint_best_val_loss_0008000_0.745783.pt`（`6.9G`）
    - `/home/darwinwsl/code/irodori/outputs/sc29_v1_full/checkpoint_best_val_loss_0010000_0.744451.pt`（`6.9G`）
  - 早期真乃实验（保留未动）：`outputs/mano_lora_baseline/`, `outputs/mano_lora_captioned/`, `outputs/mano_full_captioned/`, `outputs/mano_full_nocap/`, `outputs/mano_full_ab_noref_eval/`, `datasets/mano/`

- **Speaker Inversion Embedding 输出**
  - `/home/darwinwsl/code/irodori/outputs/sc29_v1_speakers_s8000/`：**当前 `s8000` 全量 29 角色正式目标目录**（含 `inversion_summary.json`）
  - `/home/darwinwsl/code/irodori/outputs/sc29_v1_speakers/`：早期基于 `s10000` 的 3 角色试点结果（`fuyuko.speaker.safetensors`, `mano.speaker.safetensors`, `luca.speaker.safetensors`）

- **评测报告、配置与数据**
  - 4 路微调终审评测：`/home/darwinwsl/code/irodori/outputs/sc29_v1_eval/index.html`、`summary.json`
  - 3 模式（`ref / embed / noref`）评测：`/home/darwinwsl/code/irodori/outputs/sc29_v1_eval_embed/index.html`、`summary.json`
  - 配置文件：
    - `/home/darwinwsl/code/irodori/configs/train_sc29_speaker_inversion.yaml`（`batch_size=16`, `lr=0.01`, `lr_scheduler: none`, `valid_ratio=0.05`, `valid_every=250`, `save_every=250`, `checkpoint_best_n=3`）
    - `/home/darwinwsl/code/irodori/configs/train_sc29_v1_full.yaml`
    - `/home/darwinwsl/code/irodori/configs/train_sc29_meanflow_4090.yaml`
    - `/home/darwinwsl/code/irodori/configs/train_sc29_meanflow_h100.yaml`
  - 数据与历史计划：
    - `/home/darwinwsl/code/irodori/datasets/catalog.sqlite`
    - `/home/darwinwsl/code/irodori/datasets/shared_latents/`
    - `/home/darwinwsl/code/irodori/datasets/sc29_v1/train_manifest.jsonl` & `valid_manifest.jsonl`
    - `/home/darwinwsl/code/irodori/docs/SC29_TRAINING_PLAN.md`

---

## 4. 当前 29 角色 Embedding 断点明细（基于 `model_s8000.safetensors`）

### 4.1 已完成（`3 / 29`，文件完整且已写入 `inversion_summary.json`）

| 角色 | 文件名 | 大小 | 行数 | Best Step | Best Val Loss | 耗时 |
| --- | --- | :---: | :---: | :---: | :---: | :---: |
| `amana` | `outputs/sc29_v1_speakers_s8000/amana.speaker.safetensors` | `49,264 B` | `6,251` | `500` | `0.784214` | `441.23s` |
| `asahi` | `outputs/sc29_v1_speakers_s8000/asahi.speaker.safetensors` | `49,264 B` | `4,532` | `1000` | `0.756036` | `426.33s` |
| `chiyoko` | `outputs/sc29_v1_speakers_s8000/chiyoko.speaker.safetensors` | `49,264 B` | `6,251` | `500` | `0.871167` | `437.26s` |

### 4.2 中断未完成与待跑清单（`26 / 29`）

- **中断角色（`chiyuki`）**：两次均因宿主/Agent 服务重启，恰好在跑到 `step=1000` 落盘瞬间中断（最近一次停在 `22:46`）：
  - `outputs/sc29_v1_speakers_s8000/chiyuki/` 内存在 `checkpoint_0001000.speaker.safetensors`（**`0` 字节**）及 `step 250/500/750` 检查点，顶层 `chiyuki.speaker.safetensors` 尚未生成。
- **待跑全列表（共 26 个，按字母序即 `--all-speakers` 执行顺序）**：
  `chiyuki, fuyuko, hana, haruki, hazuki, hinana, hiori, juri, kaho, kiriko, kogane, koito, luca, madoka, mamimi, mano, meguru, mei, mikoto, natsuha, nichika, rinze, sakuya, tenka, toru, yuika`
  （按单个约 `7.2 分钟`估算，余下 26 个约需 **3.1 小时**）。

### 4.3 `scripts/sc29_runner.py` 已落地的未提交加固改动

`scripts/sc29_runner.py` 较 `HEAD (4b5a26c)` 有 `+94 / -17` 行改动（已通过 `py_compile` 检查且刚跑通前序恢复逻辑）：
1. `invert-speaker` 新增 `--all-speakers`、`--skip-existing` 以及每完成一个角色即写入 `inversion_summary.json`；
2. 对任何**未跳过**的角色，启动 `train.py` 前会自动执行 `shutil.rmtree(sp_out_dir)` 清理残留脏目录（因此无需手工删 `chiyuki/`，直接续跑即可清理重来）；
3. 选 best 与复制至顶层 `<speaker>.speaker.safetensors` 前强制校验文件 `>= 40000` 字节，彻底杜绝 0 字节损坏文件被误选。

---

## 5. 新会话直接续跑指南

### 5.1 断点续跑命令（无损跳过 `amana` / `asahi` / `chiyoko`）

```bash
python3 /home/darwinwsl/code/irodori/scripts/sc29_runner.py invert-speaker \
  --all-speakers --skip-existing \
  --checkpoint /home/darwinwsl/code/irodori/outputs/sc29_v1_full/model_s8000.safetensors \
  --max-steps 1000 \
  --out-root /home/darwinwsl/code/irodori/outputs/sc29_v1_speakers_s8000
```

### 5.2 防再次断连建议（强烈推荐）

由于连续 26 个角色需要约 **3.1 小时**，且近期 IDE/Agent 后台服务曾出现过两次进程中断重启，建议二选一避免再次被打断：
1. **独立 WSL 终端托管（推荐）**：直接在原生 WSL 终端（`tmux` 或 `nohup`）运行上述命令，例如：
   ```bash
   nohup python3 /home/darwinwsl/code/irodori/scripts/sc29_runner.py invert-speaker \
     --all-speakers --skip-existing \
     --checkpoint /home/darwinwsl/code/irodori/outputs/sc29_v1_full/model_s8000.safetensors \
     --max-steps 1000 \
     --out-root /home/darwinwsl/code/irodori/outputs/sc29_v1_speakers_s8000 \
     > /tmp/sc29_invert_s8000.log 2>&1 &
   ```
   这样即使对话工具/IDE 断开，独立终端里的训练也会继续跑完；之后随时 `tail -f /tmp/sc29_invert_s8000.log` 查看即可。
2. **分批跑（每批 5–6 人，约 ~40 分钟/批）**：利用已支持的 `--speakers`（空格分隔多个角色），例如下一批先跑：
   ```bash
   python3 /home/darwinwsl/code/irodori/scripts/sc29_runner.py invert-speaker \
     --speakers chiyuki fuyuko hana haruki hazuki hinana \
     --checkpoint /home/darwinwsl/code/irodori/outputs/sc29_v1_full/model_s8000.safetensors \
     --max-steps 1000 \
     --out-root /home/darwinwsl/code/irodori/outputs/sc29_v1_speakers_s8000
   ```
   > **注意**：`invert-speaker` 中 `--speakers` 与 `--all-speakers` 互斥（见 `sc29_runner.py:345`）；如果分批用 `--speakers`，每完成一批会按本次入参覆盖刷新 `inversion_summary.json`，全部 29 人跑完后再用 `--all-speakers --skip-existing` 跑一遍（秒跳全部 29 人）即可把完整的 29 人汇总写回 `inversion_summary.json`。

### 5.3 全部 29 个角色跑完后的后续步骤

1. **验收检查**：确认 `outputs/sc29_v1_speakers_s8000/*.speaker.safetensors` 共有 `29` 个文件（每个均约 `49,264 B`），并检查 `inversion_summary.json` 中 `completed_speakers == 29`、无异常 `best_val_loss`。
2. **按需对比评测**：可用 `scripts/sc29_runner.py eval` 基于 `model_s8000.safetensors` 跑一轮 `--ref` vs `49 KB --ref-embed` 评测验证（例如输出至 `outputs/sc29_v1_eval_embed_s8000`）。
3. **归档提交**：更新 `docs/SC29_TRAINING_PLAN.md` 的 Phase 6b 完成状态，将 `scripts/sc29_runner.py` 与文档改动 commit 到 `main`。

---

## 6. Git 状态与硬性约束（新会话注意事项）

- **当前 Git 工作树**：
  - 已修改未提交：`scripts/sc29_runner.py`
  - 本交接文档：`docs/SC29_SESSION_HANDOFF.md`
  - **用户预存未跟踪文件（严禁删除或误 commit）**：`docs/AUDIO_BLIND_JUDGE_EXPERIMENT.md`、`video/`
- **当前运行环境与资源**：
  - 后台任务/残留训练进程：`0`（完全干净）；
  - GPU：RTX 4090 空闲；
  - 磁盘：`/home/darwinwsl/code/irodori` 剩余 **`589 GiB`**（使用率 `39%`）；
  - Python 注意：系统 `/usr/bin/python3` 为 **`3.8.10`**，`scripts/sc29_runner.py` 须保持 3.8 语法兼容；模型训练由脚本自动调起 `Irodori-TTS/.venv/bin/python`，且已内置 `HF_HUB_OFFLINE=1` 与 `tokenizer` 软链处理。
- **保留与改动红线**：
  1. **严禁删除**：`outputs/mano_*`、`datasets/mano/`、`outputs/sc29_v1_full/kept/*`、`outputs/sc29_v1_full/checkpoint_best_val_loss_0010000_0.744451.pt`；
  2. **保持 Submodule 干净**：不修改 `Irodori-TTS/` 与 `voice-caption-annotator/` 目录内容；
  3. **不擅自开启 MeanFlow 蒸馏**：除非用户明确要求恢复 Phase 7，否则不运行蒸馏训练。
