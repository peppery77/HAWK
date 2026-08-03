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

## 重新复现结果

以下结果由本仓库代码使用 VLMEvalKit、原生动态分辨率和 L2 归一化重新运行得到：

| 数据集 | 视觉 Token 剪枝率 | 论文结果 | 重新复现 | 差值 |
|---|---:|---:|---:|---:|
| ChartQA | 60% | 83.600 | **85.160** | +1.560 |
| ChartQA | 80% | 76.800 | **79.040** | +2.240 |
| ChartQA | 90% | 65.200 | **67.440** | +2.240 |
| TextVQA | 60% | **85.000** | **85.000** | +0.000 |
| TextVQA | 80% | 83.000 | **83.156** | +0.156 |
| TextVQA | 90% | **79.800** | 79.152 | -0.648 |
| RealWorldQA | 60% | 67.600 | **67.712** | +0.112 |
| RealWorldQA | 80% | **65.000** | 62.353 | -2.647 |
| RealWorldQA | 90% | **60.400** | 60.000 | -0.400 |

表中“重新复现”不是论文数值的转录。全部实验使用原始 HAWK 注意力头重要性向量以及修正后的 generation state。公开代码中的权重仅统一缩放至和为 1，不会改变 top-k Token 选择。

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

make setup PYTHON_BIN="$(which python)"
make test PYTHON_BIN="$(which python)"
```

`make setup` 只修改 `.runtime/python` 中的项目局部 Transformers，不会修改 Conda 环境或系统全局安装。

### 2. 从 Hugging Face 下载模型

```bash
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

下面一个命令会自动准备 VLMEvalKit、从 Hugging Face 下载所选数据集、转换数据格式并完成推理与评分：

```bash
python scripts/evaluate.py \
  --task realworldqa \
  --pruning_ratio 0.8 \
  --model_path models/Qwen2.5-VL-7B-Instruct \
  --gpus 0,1,2,3
```

`pruning_ratio` 表示被删除的视觉 Token 比例。支持的 `task` 为 `realworldqa`、`chartqa`、`textvqa`、`mme` 和 `scienceqa`。无剪枝基线设为 `0`，论文原生分辨率实验分别设为 `0.6`、`0.8` 和 `0.9`。

一次运行三项数据集和三个剪枝率：

```bash
for task in chartqa textvqa realworldqa; do
  for pruning_ratio in 0.6 0.8 0.9; do
    python scripts/evaluate.py \
      --task "${task}" \
      --pruning_ratio "${pruning_ratio}" \
      --gpus 0,1,2,3
  done
done
```

原生动态分辨率使用 `min_pixels=1,003,520`、`max_pixels=12,845,056`：

| 论文设置 | `pruning_ratio` |
|---|---:|
| 无剪枝 | `0.0` |
| 原生分辨率剪枝 60% | `0.6` |
| 原生分辨率剪枝 80% | `0.8` |
| 原生分辨率剪枝 90% | `0.9` |

数据集与模型均使用 Hugging Face 官方客户端及标准缓存。预测、得分、运行清单和逐样本剪枝统计保存在 `vlmeval_results/<RUN_NAME>/`。完整的 Python API 和兼容范围见[英文 README](README.md)。

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
