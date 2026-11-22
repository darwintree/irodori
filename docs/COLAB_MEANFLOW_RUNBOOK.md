# Colab G4 (RTX PRO 6000 96 GB) MeanFlow 4-Step 蒸馏运行手册

本文档面向在 Google Colab 中启动的 **`agy cli`**（以及人工操作），提供基于 `model_s8000.safetensors` 的极简 MeanFlow 4-step 蒸馏与跨 24 小时断点续训标准流程。

---

## 1. 路径约定与核心原则

- **底座一致**：Teacher 固定使用 **`model_s8000.safetensors`**，保证冻结的 `speaker_encoder` / 条件投影层与本地训练好的 29 个 `s8000` 49 KB `--ref-embed` 完全兼容。
- **Google Drive 目录（持久存储）**：
  - 输入包：`/content/drive/MyDrive/irodori_sc29/bundle/`
  - 断点与导出备份：`/content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/`
- **Colab 本地工作区（NVMe 高速盘，跨会话路径必须固定，否则 `--resume` 会拒绝恢复）**：
  - 根目录：`/content/sc29_work/`
  - Teacher：`/content/sc29_work/teacher/model_s8000.safetensors`
  - Manifest：`/content/sc29_work/manifests/train_manifest_colab.jsonl`
  - 训练输出：`/content/sc29_work/outputs/sc29_v1_meanflow/`
  - Speaker Embeddings：`/content/sc29_work/speakers_s8000/`
- **硬性铁律**：
  1. **绝不把 `train.py --output-dir` 直接设在 `/content/drive/...`**（MeanFlow `.pt` 单文件约 3–5 GB，直写 Drive FUSE 极慢且断连易损坏）。训练全部写 `/content/sc29_work/outputs/...`，再由 `agy cli` 在命令行周期性原子拷贝到 Drive。
  2. **不要把 15.4 万个散装 `.pt` 放 Drive 上直接读**：统一走 `shared_latents.tar` 拷贝到本地 NVMe 后解包。

---

## 2. Step 0：本机 WSL 一次性打包上传（29 角色 Inversion 完成后执行）

在本机 `/home/darwinwsl/code/irodori` 运行：

```bash
python3 scripts/colab_prepare_meanflow.py pack
```

产物默认输出到 `outputs/colab_meanflow_bundle/`（约 `5.8 GiB`，含 `model_s8000.safetensors`、`tokenizer.tar`、`shared_latents.tar`、`manifests/`、`speakers_s8000.tar`）。将其全部上传至 Google Drive 的 `MyDrive/irodori_sc29/bundle/`。

---

## 3. Step 1：Colab 环境与 Blackwell (`sm_120`) 校验

推荐直接通过仓库中的 `notebooks/colab_meanflow_setup.ipynb` 完成 Drive 挂载、`Irodori-TTS`（检出测试基线 `f01c08c`）克隆并执行 `uv sync --extra cu128`。该命令会在 `Irodori-TTS/.venv/` 中安装 `torch>=2.10.0,<2.11.0` + `cu128`，原生支持 Blackwell (`sm_120`)。

校验目录、GPU 与 PyTorch (`Irodori-TTS/.venv/bin/python`) 可用性：

```bash
ls -lh /content/drive/MyDrive/irodori_sc29/bundle/
Irodori-TTS/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x=torch.randn(8,8,device='cuda'); print((x@x).sum().item())"
```

- **依赖安装注意**：RTX PRO 6000（Blackwell `sm_120`）需要支持 `sm_120` 的 PyTorch + CUDA 12.8（`uv sync --extra cu128` 默认满足）；后续所有训练、转换与推理统一使用 `Irodori-TTS/.venv/bin/python`，不要与系统 Python 混装降级版本。
- 确认 `HF_HUB_OFFLINE=0` 可正常访问 HuggingFace 拉取 `sbintuitions/modernbert-ja-310m` 与 `Aratako/Semantic-DACVAE-Japanese-32dim`（或已缓存到本地）。

---

## 4. Step 2：解包到本地 NVMe 并重写 Manifest

在仓库根目录执行：

```bash
python3 scripts/colab_prepare_meanflow.py unpack \
  --bundle-dir /content/drive/MyDrive/irodori_sc29/bundle \
  --work-dir /content/sc29_work
```

脚本会自动：
1. 复制 `model_s8000.safetensors` 到 `/content/sc29_work/teacher/model_s8000.safetensors`；
2. 解包 `tokenizer.tar` 并在 `/content/sc29_work/outputs/sc29_v1_meanflow/tokenizer` 建立软链；
3. 解包 `shared_latents.tar` 到 `/content/sc29_work/shared_latents/`；
4. 生成 `/content/sc29_work/manifests/train_manifest_colab.jsonl` 并校验前 5 条 `latent_path` 存在；
5. 解包 `speakers_s8000.tar` 到 `/content/sc29_work/speakers_s8000/`。

---

## 5. Step 3：10 步冒烟测速（强烈建议）

正式开跑前先用 `/tmp/mf_probe` 测 10 步速度与显存：

```bash
IRODORI_SDPA_CUDNN=0 Irodori-TTS/.venv/bin/python Irodori-TTS/train.py \
  --config configs/train_sc29_meanflow_g4_colab.yaml \
  --manifest /content/sc29_work/manifests/train_manifest_colab.jsonl \
  --output-dir /tmp/mf_probe \
  --max-steps 10 \
  --log-every 2 \
  --save-every 999999 \
  --checkpoint-best-n 0 \
  --valid-ratio 0.0 \
  --valid-every 0
```

- 正常显存与速度无问题后清理 `rm -rf /tmp/mf_probe`。
- 若单步较慢且需要开启 `torch.compile`（`compile_model: true` / `meanflow_compile_teacher: true`），先在 probe 中验证 `sm_120` 下无 Triton 报错再修改 `configs/train_sc29_meanflow_g4_colab.yaml`（注意：跨会话 `--resume` 也允许更改 compile 与 `meanflow_teacher_chunk_size` 设置）。

---

## 6. Step 4：启动或跨 24 小时断点续训

先检查 Drive 是否已有历史断点：

```bash
mkdir -p /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow
ls -lh /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/
```

### 情况 A：首次启动（Drive 中无 `.pt`）

```bash
IRODORI_SDPA_CUDNN=0 Irodori-TTS/.venv/bin/python Irodori-TTS/train.py \
  --config configs/train_sc29_meanflow_g4_colab.yaml \
  --manifest /content/sc29_work/manifests/train_manifest_colab.jsonl \
  --output-dir /content/sc29_work/outputs/sc29_v1_meanflow \
  > /content/sc29_work/meanflow_train.log 2>&1
```

### 情况 B：新 Colab 会话续训（Drive 中已有 `.pt`）

先把 Drive 里的最新周期 `.pt`（以及已保存的 `checkpoint_best_val_loss_*.pt`）拷回本地输出目录，再带 `--resume` 启动：

```bash
cp -v /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/*.pt \
  /content/sc29_work/outputs/sc29_v1_meanflow/

# 将 <LATEST_CKPT.pt> 替换为最新 step 的 checkpoint 文件名
IRODORI_SDPA_CUDNN=0 Irodori-TTS/.venv/bin/python Irodori-TTS/train.py \
  --config configs/train_sc29_meanflow_g4_colab.yaml \
  --manifest /content/sc29_work/manifests/train_manifest_colab.jsonl \
  --output-dir /content/sc29_work/outputs/sc29_v1_meanflow \
  --resume /content/sc29_work/outputs/sc29_v1_meanflow/<LATEST_CKPT.pt> \
  >> /content/sc29_work/meanflow_train.log 2>&1
```

---

## 7. Step 5：训练期间 `agy cli` 盯盘与原子备份到 Drive

`configs/train_sc29_meanflow_g4_colab.yaml` 默认设了 **`save_every: 500`**、`checkpoint_best_n: 2`（`train.py` 本地会自动只保留最新 1 份周期 `.pt` + 最优 2 份 `best_val_loss` `.pt`）。

训练运行期间，由 `agy cli` 在命令行周期性（建议每 1–2 小时，也可配合 `/schedule`）检查一次：

1. 看进度与最新落盘文件：
   ```bash
   tail -n 20 /content/sc29_work/meanflow_train.log | tr '\r' '\n' | tail -n 15
   ls -lht /content/sc29_work/outputs/sc29_v1_meanflow/*.pt
   ```
2. 确认新 `.pt` 写入完成（文件非 0 且大小稳定 / 日志已打印下一轮 step）后，用 **临时文件 + `mv` 原子改名** 备份到 Drive，同时同步日志：
   ```bash
   DRIVE_CKPT=/content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow
   for f in /content/sc29_work/outputs/sc29_v1_meanflow/*.pt; do
     name=$(basename "$f")
     if [ ! -f "$DRIVE_CKPT/$name" ]; then
       cp "$f" "$DRIVE_CKPT/$name.tmp" && mv "$DRIVE_CKPT/$name.tmp" "$DRIVE_CKPT/$name"
     fi
   done
   cp /content/sc29_work/meanflow_train.log "$DRIVE_CKPT/meanflow_train.log.tmp" \
     && mv "$DRIVE_CKPT/meanflow_train.log.tmp" "$DRIVE_CKPT/meanflow_train.log"
   ```
3. **清理 Drive 旧周期断点**：只保留 Drive 上 **最新 1–2 个周期 `checkpoint_00*.pt`** + **最优 2 个 `checkpoint_best_val_loss_*.pt`**，删除更早的旧周期文件以节省 Drive 空间。
4. **接近 24 小时断连前**：等到最近一次 `save_every=500` 落盘并完成上述原子备份后，手动结束训练进程等待下一轮 Colab 会话续跑。

---

## 8. Step 6：蒸馏完成导出与 4-step + `--ref-embed` 冒烟验证

1. 挑选最终最稳的 `.pt`（通常是 `checkpoint_best_val_loss_*.pt` 或 `checkpoint_final.pt`），导出为推理专用 `.safetensors`（自动包含 `tokenizer/` 与 `flow_parameterization: meanflow` 元数据）：
   ```bash
   Irodori-TTS/.venv/bin/python Irodori-TTS/convert_checkpoint_to_safetensors.py \
     /content/sc29_work/outputs/sc29_v1_meanflow/<BEST_OR_FINAL.pt> \
     --output /content/sc29_work/outputs/sc29_v1_meanflow_export/model_meanflow_s8000.safetensors \
     --force
   ```
2. 抽取 1 个 `s8000` 49 KB `--ref-embed` 跑 4-step 冒烟合成（MeanFlow 默认 4 步，忽略 RF CFG 参数）：
   ```bash
   Irodori-TTS/.venv/bin/python Irodori-TTS/infer.py \
     --checkpoint /content/sc29_work/outputs/sc29_v1_meanflow_export/model_meanflow_s8000.safetensors \
     --ref-embed /content/sc29_work/speakers_s8000/fuyuko.speaker.safetensors \
     --text "プロデューサーさん、今日のお仕事もふゆにお任せくださいねっ！" \
     --caption "猫かぶりな営業口調で、甘く可愛らしい高い声で明るく話している。" \
     --output-wav /content/sc29_work/smoke_fuyuko_mf4.wav
   ```
3. 将导出的 `/content/sc29_work/outputs/sc29_v1_meanflow_export/`（含 `model_meanflow_s8000.safetensors` 和 `tokenizer/`）打包回传 Drive 供本机下载评测：
   ```bash
   tar -cf /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/model_meanflow_s8000_export.tar.tmp \
     -C /content/sc29_work/outputs/sc29_v1_meanflow_export . \
     && mv /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/model_meanflow_s8000_export.tar.tmp \
           /content/drive/MyDrive/irodori_sc29/checkpoints/sc29_v1_meanflow/model_meanflow_s8000_export.tar
   ```
