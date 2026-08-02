PYTHON_BIN ?= python3
MODEL_PATH ?= models/Qwen2.5-VL-7B-Instruct
IMAGE ?= assets/smoke_test.png
GPU ?= cuda:1
KEEP_RATIO ?= 0.198

.PHONY: setup setup-vlmeval prepare-vlmeval download test demo smoke-image \
	vlmeval-baseline vlmeval-hawk

setup:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_runtime.sh

setup-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_vlmevalkit.sh

prepare-vlmeval:
	HF_ENDPOINT=https://hf-mirror.com PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh \
		scripts/prepare_vlmeval_datasets.py --project-root .

download:
	HF_ENDPOINT=https://hf-mirror.com PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh \
		scripts/download_model.py --local-dir "$(MODEL_PATH)"

test:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh -m pytest

smoke-image:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/create_smoke_image.py --output "$(IMAGE)"

demo: smoke-image
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh scripts/infer.py \
		--model-path "$(MODEL_PATH)" \
		--image "$(IMAGE)" \
		--device "$(GPU)" \
		--keep-ratio "$(KEEP_RATIO)"

vlmeval-baseline:
	PYTHON_BIN="$(PYTHON_BIN)" DATASETS=RealWorldQA KEEP_RATIO=1.0 \
		RUN_NAME=vlmeval_native_baseline \
		scripts/evaluate_vlmeval.sh

vlmeval-hawk:
	PYTHON_BIN="$(PYTHON_BIN)" DATASETS=RealWorldQA KEEP_RATIO=0.20 \
		RUN_NAME=vlmeval_native_hawk_p80 \
		scripts/evaluate_vlmeval.sh
