<div align="center">

# HAWK

### Head Importance-Aware Visual Token Pruning in Multimodal Models

**CVPR 2026**

[![Paper](https://img.shields.io/badge/arXiv-2604.07812-b31b1b.svg)](https://arxiv.org/abs/2604.07812)
[![Conference](https://img.shields.io/badge/CVPR-2026-4b44ce.svg)](https://cvpr.thecvf.com/)
[![Model](https://img.shields.io/badge/Model-Qwen2.5--VL--7B-0f766e.svg)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**Training-free visual token pruning that combines static head importance with dynamic text-guided relevance.**

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## Overview

Visual tokens dominate the context length of multimodal large language models. HAWK removes redundant visual tokens before the full language-model forward pass while preserving tokens that matter for the current instruction. It requires no training or fine-tuning.

<p align="center">
  <img src="assets/method.png" alt="HAWK method overview" width="96%">
</p>

HAWK has three ingredients:

1. **Static visual-head importance** estimated offline through head ablation.
2. **Dynamic text-guided relevance** computed from first-layer text-to-vision QK scores without RoPE.
3. **Importance-aware top-k pruning** that aggregates per-head scores and preserves the original order and multimodal position IDs of retained tokens.

## Highlights

- **Training-free and plug-and-play:** no additional parameters or fine-tuning.
- **High retention:** preserves **96.0%** average relative performance on Qwen2.5-VL-7B after pruning **80.2%** of visual tokens at fixed resolution.
- **Native-resolution support:** preserves **99.6%**, **96.2%**, and **89.7%** relative performance at 60%, 80%, and 90% pruning.
- **Practical efficiency:** at 80% pruning on MME, reduces end-to-end latency from 20m15s to 15m04s (**1.34x** speedup), KV cache from 668 MB to 148 MB, and peak memory from 16.9 GB to 15.7 GB.
- **Correct generation state:** synchronizes input IDs, masks, mRoPE positions, cache positions, and `rope_deltas` after prompt compaction.

## Results

Native-resolution Qwen2.5-VL-7B results from Table 3 of the paper:

| Visual pruning | MME | TextVQA | ChartQA | RealWorldQA | Average relative performance |
|---:|---:|---:|---:|---:|---:|
| 0% | 2315.0 | 85.2 | 86.2 | 67.7 | 100.0% |
| 60% | **2313.0** | **85.0** | **83.6** | **67.6** | **99.6%** |
| 80% | **2311.0** | **83.0** | **76.8** | **65.0** | **96.2%** |
| 90% | **2101.0** | **79.8** | **65.2** | **60.4** | **89.7%** |

<p align="center">
  <img src="assets/cvpr2026_radar.png" alt="HAWK benchmark comparison" width="72%">
</p>

## Quick Start

The commands below start from a clean Linux clone and use Qwen2.5-VL-7B-Instruct. The implementation has been validated with Python 3.10, PyTorch 2.5, Transformers 4.52.0, CUDA, and A100 40 GB GPUs.

### 1. Clone and create the environment

```bash
git clone https://github.com/peppery77/HAWK.git
cd HAWK

conda create -n hawk python=3.10 -y
conda activate hawk

# Choose the PyTorch build that matches your CUDA installation.
pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps
```

For users in China, the Python package index can be changed before setup:

```bash
export HAWK_PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
```

### 2. Build the isolated patched runtime

HAWK patches a project-local copy of Transformers 4.52.0 under `.runtime/python`; it does not modify the global or Conda installation.

```bash
make setup PYTHON_BIN="$(which python)"
make test PYTHON_BIN="$(which python)"
```

### 3. Download Qwen2.5-VL-7B from HF Mirror

```bash
HF_ENDPOINT=https://hf-mirror.com \
make download PYTHON_BIN="$(which python)"
```

The checkpoint is stored at `models/Qwen2.5-VL-7B-Instruct/` and is excluded from Git.

### 4. Run the smoke test

```bash
make demo \
  PYTHON_BIN="$(which python)" \
  GPU=cuda:0 \
  KEEP_RATIO=0.198
```

This creates a deterministic redistributable test image and prints JSON containing the generated answer, elapsed time, peak GPU memory, original visual-token count, retained count, and actual retention ratio.

### 5. Run a custom image

```bash
scripts/run.sh scripts/infer.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --image /absolute/path/to/image.jpg \
  --prompt "Describe this image in detail." \
  --keep-ratio 0.198 \
  --device cuda:0
```

## Reproduce with VLMEvalKit

The paper uses VLMEvalKit. This repository pins the compatible `ms-vlmeval==0.0.18` package, copies it into `.runtime/VLMEvalKit`, and applies a narrow Qwen2.5-VL attention-backend patch.

### 1. Prepare VLMEvalKit and RealWorldQA

```bash
make setup-vlmeval PYTHON_BIN="$(which python)"

HF_ENDPOINT=https://hf-mirror.com \
scripts/run.sh scripts/prepare_vlmeval_datasets.py \
  --project-root . \
  --datasets RealWorldQA
```

To prepare both supported MCQ datasets, pass `--datasets RealWorldQA ScienceQA_TEST`.

### 2. Run baseline and HAWK

Single GPU:

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 DATASETS=RealWorldQA \
KEEP_RATIO=1.0 RUN_NAME=realworldqa_baseline \
scripts/evaluate_vlmeval.sh

CUDA_DEVICES=0 NUM_PROCESSES=1 DATASETS=RealWorldQA \
KEEP_RATIO=0.20 RUN_NAME=realworldqa_hawk_p80 \
scripts/evaluate_vlmeval.sh
```

Five GPUs:

```bash
CUDA_DEVICES=0,1,2,3,4 NUM_PROCESSES=5 DATASETS=RealWorldQA \
KEEP_RATIO=0.40 RUN_NAME=realworldqa_hawk_p60 \
scripts/evaluate_vlmeval.sh
```

`KEEP_RATIO` is the fraction retained, not the fraction removed. Each run writes its generated config, prediction files, accuracy files, manifest, and per-sample pruning traces to `vlmeval_results/<RUN_NAME>/`.

### Native-resolution settings

The evaluation path preserves aspect ratio and uses:

```text
min_pixels = 1280 × 28 × 28 = 1,003,520
max_pixels = 16384 × 28 × 28 = 12,845,056
```

| Paper setting | `KEEP_RATIO` | Meaning |
|---|---:|---|
| Baseline | `1.0` | Retain all visual tokens |
| Native 60% pruning | `0.40` | Retain 40% |
| Native 80% pruning | `0.20` | Retain 20% |
| Native 90% pruning | `0.10` | Retain 10% |
| Fixed 60.5% pruning | `0.395` | Retain 39.5% |
| Fixed 80.2% pruning | `0.198` | Retain 19.8% |
| Fixed 90.1% pruning | `0.099` | Retain 9.9% |

Selection uses `ceil(num_visual_tokens × keep_ratio)` and always retains at least one visual token.

### Additional datasets

The local adapters currently support `RealWorldQA`, `ScienceQA_TEST`, `ChartQA_TEST`, `TextVQA_VAL`, and `MME`.

```bash
# TextVQA validation (5,000 questions)
scripts/run.sh scripts/prepare_textvqa_hf.py --project-root .

# MME (2,374 questions)
scripts/run.sh scripts/prepare_mme_hf.py --project-root .

# ChartQA test split
scripts/run.sh scripts/prepare_chartqa_hf.py --project-root .
```

Then set `DATASETS` to the corresponding dataset name. For very large native-resolution samples, `evaluate_vlmeval.sh` enables `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator fragmentation.

## Head Weights and Scoring

For each attention head `i`, HAWK computes first-layer projections without RoPE:

```text
Q_i = H_text W_q_i
K_i = H_visual W_k_i
A_i = Q_i K_i^T / sqrt(d_k)
```

The implementation averages over text queries, L2-normalizes each head's visual score vector, and computes:

```text
score(token_k) = Σ_i weight_i × normalized_score_i(token_k)
```

The released 28-head vector is normalized to sum to 1. The raw ablation vector is retained as `QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY`; dividing every entry by the same positive constant preserves the exact top-k ranking mathematically.

Alternative experiments can pass a 28-value JSON array and normalization mode:

```bash
HEAD_WEIGHTS_JSON='[...28 values...]' \
SCORE_NORMALIZATION=l2 \
CUDA_DEVICES=0 DATASETS=RealWorldQA KEEP_RATIO=0.20 \
RUN_NAME=custom_weights scripts/evaluate_vlmeval.sh
```

## Python API

```python
from hawk import HawkConfig, configure_model, get_last_pruning_stats

configure_model(model, HawkConfig(keep_ratio=0.198))
generated_ids = model.generate(**inputs, max_new_tokens=64, num_beams=1)
stats = get_last_pruning_stats(model)
```

For a complete example, see [`scripts/infer.py`](scripts/infer.py).

## Repository Layout

```text
src/hawk/config.py              ratios and normalized head weights
src/hawk/pruning.py             scoring, normalization, and ordered top-k
src/hawk/integration.py         public model configuration API
src/hawk/vlmeval_adapter.py     VLMEvalKit model and dataset adapters
scripts/patch_transformers.py   Qwen2.5-VL + generation-state integration
scripts/infer.py                end-to-end image inference
scripts/evaluate_vlmeval.sh     distributed native-resolution evaluation
tests/                          unit and patch-idempotency tests
```

## Compatibility

- Qwen2.5-VL-7B-Instruct with Transformers 4.52.0.
- Inference only; training losses are intentionally unsupported.
- Batch size 1 and `num_beams=1` on the pruning path.
- SDPA attention backend.
- Image and video placeholder spans are recognized; the included public benchmark scripts cover image evaluation.

## Citation

```bibtex
@inproceedings{zhu2026hawk,
  title     = {HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models},
  author    = {Zhu, Qihui and Zhang, Tao and Wang, Yuchen and Wen, Zijian and
               Zhang, Mengjie and Chen, Shuangwu and Tan, Xiaobin and Yang, Jian and
               Liu, Yang and Dong, Zhenhua and Yu, Xianzhi and Pan, Yinfei},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2026}
}
```

## License

Released under the [Apache License 2.0](LICENSE). Qwen2.5-VL, Transformers, VLMEvalKit, and benchmark datasets remain subject to their respective licenses and terms.
