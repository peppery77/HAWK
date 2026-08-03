PYTHON_BIN ?= python3
MODEL_PATH ?= models/Qwen2.5-VL-7B-Instruct
IMAGE ?= assets/smoke_test.png
GPU ?= cuda:1
KEEP_RATIO ?= 0.198
TASK ?= realworldqa
PRUNING_RATIO ?= 0.8
GPUS ?= 0

.PHONY: setup setup-vlmeval prepare-vlmeval download test demo smoke-image evaluate

setup:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_runtime.sh

setup-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/setup_vlmevalkit.sh

prepare-vlmeval:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh \
		scripts/prepare_vlmeval_datasets.py --project-root .

download:
	PYTHON_BIN="$(PYTHON_BIN)" scripts/run.sh \
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

evaluate:
	"$(PYTHON_BIN)" scripts/evaluate.py \
		--task "$(TASK)" \
		--pruning_ratio "$(PRUNING_RATIO)" \
		--model_path "$(MODEL_PATH)" \
		--gpus "$(GPUS)"
