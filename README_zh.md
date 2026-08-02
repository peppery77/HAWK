<div align="center">

# HAWK

### 面向多模态大模型的注意力头重要性感知视觉 Token 剪枝

**CVPR 2026**

[![论文](https://img.shields.io/badge/arXiv-2604.07812-b31b1b.svg)](https://arxiv.org/abs/2604.07812)
[![开源协议](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

**无需训练，将静态注意力头重要性与动态文本引导相关性相结合。**

[English](README.md) | [简体中文](README_zh.md)

</div>

---

## 简介

HAWK 在完整语言模型前向传播之前移除冗余视觉 Token，同时保留与当前文本指令最相关的视觉信息。方法包含三个部分：离线得到的静态注意力头重要性、第一层 LLM 中去除 RoPE 后的文本到视觉 QK 分数，以及保持原始 Token 顺序和多模态位置编码的 top-k 剪枝。

<p align="center">
  <img src="assets/method.png" alt="HAWK 方法概览" width="96%">
</p>

论文在 Qwen2.5-VL-7B 上剪除 80.2% 视觉 Token 后仍保持 96.0% 的平均相对性能。在原生动态分辨率下，60%、80% 和 90% 剪枝分别保持 99.6%、96.2% 和 89.7% 的平均相对性能。

## 快速开始

以下流程面向 Linux + CUDA 环境，已验证版本为 Python 3.10、PyTorch 2.5、Transformers 4.52.0 和 A100 40 GB。

### 1. 创建环境

```bash
git clone https://github.com/peppery77/HAWK.git
cd HAWK

conda create -n hawk python=3.10 -y
conda activate hawk

pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e . --no-deps

export HAWK_PIP_INDEX_URL=https://mirrors.ustc.edu.cn/pypi/simple
make setup PYTHON_BIN="$(which python)"
make test PYTHON_BIN="$(which python)"
```

`make setup` 只修改 `.runtime/python` 中的项目局部 Transformers，不会修改 Conda 环境或系统全局安装。

### 2. 通过 HF Mirror 下载模型

```bash
HF_ENDPOINT=https://hf-mirror.com \
make download PYTHON_BIN="$(which python)"
```

### 3. 运行单图测试

```bash
make demo \
  PYTHON_BIN="$(which python)" \
  GPU=cuda:0 \
  KEEP_RATIO=0.198
```

自定义图片：

```bash
scripts/run.sh scripts/infer.py \
  --model-path models/Qwen2.5-VL-7B-Instruct \
  --image /绝对路径/图片.jpg \
  --prompt "请详细描述这张图片。" \
  --keep-ratio 0.198 \
  --device cuda:0
```

输出 JSON 包含生成结果、耗时、峰值显存、剪枝前后 Token 数和实际保留率。

## 使用 VLMEvalKit 复现

### 1. 准备 VLMEvalKit 和 RealWorldQA

```bash
make setup-vlmeval PYTHON_BIN="$(which python)"

HF_ENDPOINT=https://hf-mirror.com \
scripts/run.sh scripts/prepare_vlmeval_datasets.py \
  --project-root . \
  --datasets RealWorldQA
```

### 2. 运行 80% 剪枝评测

```bash
CUDA_DEVICES=0 NUM_PROCESSES=1 DATASETS=RealWorldQA \
KEEP_RATIO=0.20 RUN_NAME=realworldqa_hawk_p80 \
scripts/evaluate_vlmeval.sh
```

多卡 60% 剪枝示例：

```bash
CUDA_DEVICES=0,1,2,3,4 NUM_PROCESSES=5 DATASETS=RealWorldQA \
KEEP_RATIO=0.40 RUN_NAME=realworldqa_hawk_p60 \
scripts/evaluate_vlmeval.sh
```

`KEEP_RATIO` 表示保留比例，而不是剪除比例。原生动态分辨率使用 `min_pixels=1,003,520`、`max_pixels=12,845,056`：

| 论文设置 | `KEEP_RATIO` |
|---|---:|
| 无剪枝 | `1.0` |
| 原生分辨率剪枝 60% | `0.40` |
| 原生分辨率剪枝 80% | `0.20` |
| 原生分辨率剪枝 90% | `0.10` |
| 固定分辨率剪枝 80.2% | `0.198` |

预测、得分、运行清单和逐样本剪枝统计保存在 `vlmeval_results/<RUN_NAME>/`。

当前本地适配器支持 RealWorldQA、ScienceQA、ChartQA、TextVQA 和 MME。完整的数据准备命令、Python API、兼容范围和论文结果表见[英文 README](README.md)。

## 权重说明

公开的 28 个注意力头权重按论文公式进行 L1 归一化，权重和为 1。原始向量保留在 `QWEN2_5_VL_7B_HEAD_WEIGHTS_LEGACY` 中；对全部权重统一除以正数不会改变 top-k 排序，因此不会改变保留的视觉 Token。

## 引用

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

## 开源协议

本项目采用 [Apache License 2.0](LICENSE)。Qwen2.5-VL、Transformers、VLMEvalKit 和各评测数据集仍遵循各自的许可证及使用条款。
