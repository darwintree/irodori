# Irodori-TTS Workspace (Shiny Colors)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darwintree/irodori/blob/main/notebooks/colab_meanflow_setup.ipynb)

Workspace for preparing Shiny Colors voice datasets, training, and evaluating [Aratako/Irodori-TTS](https://github.com/Aratako/Irodori-TTS) (`v4.1-Small`) on RTX 4090 (24 GB VRAM, WSL2).

## Directory Structure

### Tracked in Git
- `configs/` — Training YAML configs (e.g. `train_mano_lora_baseline.yaml`, `train_sc29_meanflow_g4_colab.yaml`).
- `notebooks/` — Colab setup notebooks:
  - `colab_meanflow_setup.ipynb` — Mounts Drive, clones workspace + `Irodori-TTS` (`f01c08c`), runs `uv sync --extra cu128`, and unpacks the SC29 MeanFlow bundle.
- `scripts/` — Data export, Colab packaging, and local monitoring/evaluation tools:
  - `prepare_mano_wavs.py` — Converts `.m4a` voice clips to 48 kHz mono PCM WAV with leading/trailing silence trimmed (`silenceremove`, keeping `0.15s` margin).
  - `monitor_server.py` — Lightweight HTTP dashboard (`:8765`) for loss curves and multi-sentence × multi-caption audio evaluation.
  - `colab_prepare_meanflow.py` — Packs/unpacks SC29 MeanFlow assets and rewrites manifest paths for Colab.
- `docs/COLAB_MEANFLOW_RUNBOOK.md` — Runbook for `agy cli` to execute MeanFlow distillation and Google Drive checkpoint backups in Colab.
- `README.md`, `.gitignore`

### Ignored by Git (Local / Large Artifacts)
- `Irodori-TTS/` — Upstream repository clone (pinned tested commit: `f01c08c`) and `uv` virtualenv (`.venv`).
- `models/` — Pretrained weights downloaded from Hugging Face (`Aratako/Irodori-TTS-v4.1-Small`).
- `datasets/` — Exported WAVs, DACVAE latents, and `train_manifest.jsonl`.
- `outputs/` — Training checkpoints, run configs, and generated `eval_samples/`.

## Environment Setup

1. Clone upstream repo and install dependencies (CUDA 12.8):
   ```bash
   git clone https://github.com/Aratako/Irodori-TTS.git
   cd Irodori-TTS && uv sync --extra cu128
   ```
2. Download base model into `models/Irodori-TTS-v4.1-Small`:
   ```bash
   hf download Aratako/Irodori-TTS-v4.1-Small --local-dir models/Irodori-TTS-v4.1-Small
   ```

## Critical Runtime Flags (RTX 4090 / Variable-Length Inputs)

Always set these environment variables when running `train.py` or `infer.py`:

```bash
export IRODORI_SDPA_CUDNN=0
export TOKENIZERS_PARALLELISM=false
```

- **Why `IRODORI_SDPA_CUDNN=0`?** `SDPBackend.CUDNN_ATTENTION` recompiles execution plans whenever sequence length varies across batches (`~5.6s/step` overhead vs `~0.019s` with `EFFICIENT_ATTENTION`). Disabling cuDNN SDPA speeds up training on RTX 4090 from `~7–19s/step` to `~1.3–1.5 step/s`.

## Data Source Notes

- Source metadata lives in MongoDB `shinycolors.voices` (with `duration` populated and indexed), audio files under `/mnt/p/sc/sc-asset-map-manager/assets/sounds/voice/events/`.
- Always export working WAVs and latents onto native Linux ext4 (`datasets/`) rather than training directly over `/mnt/p/`.
- Recommended duration window: `3–15s` (48 kHz mono PCM, silence-trimmed, normalized to `-16 dB` during `prepare_manifest.py`).

## Phase 1 Baseline Summary

- **Target**: Single-character (`真乃`), caption-free LoRA baseline (`lora_r=16`, `lora_alpha=32`, `batch_size=8`, `grad_accum=4`, `max_latent_steps=500`, `ref_max_seconds=30.0`).
- **Dataset**: `5,549` clips (`8.01h`) in `datasets/mano/`.
- **Training**: Stopped at step `3,000` (`outputs/mano_lora_baseline/checkpoint_0003000`). Validation loss improved from `0.7583` (@500) to `0.7349` (@3000); train `dur_mae` dropped from `12.58` to `3.73` frames.
- **Evaluation**: Compared `base_zero_shot` vs `step_0003000` on a 15-item no-emoji multi-sentence × multi-caption matrix (including ambiguous text with contrasting captions, text-caption emotion mismatch, long text, short-ref, and `--no-ref`).
- **Takeaway**: Overall listening quality / improvement over zero-shot is not yet satisfactory; workspace initialized here before redesigning the next training/data/caption strategy.
