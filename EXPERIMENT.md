# Experiment Instructions

This document records the exact steps used to reproduce the COMP 468 H2O experiments.

## 1. Clone Repository

Original repository:

```bash
git clone https://github.com/FMInference/H2O.git
```

My version:

```bash
git clone https://github.com/vwang1103/H2O.git
```

## 2. Create Environment

```bash
conda create -n h2o python=3.10 -y
conda activate h2o
```

## 3. Install and Fix Dependencies

From the repository README:

```bash
pip install torch
pip install git+https://github.com/huggingface/transformers
pip install crfm-helm
pip install lm-eval
```

My dependency fixes:

```bash
# 1. PyTorch matching the host CUDA driver
#    Driver 565.x supports CUDA 12.7; use cu121 wheels.
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.4.1" "torchvision==0.19.1"

# 2. Transformers pinned to the version H2O was developed against.
pip install "transformers==4.28.1"

# 3. Legacy lm-eval is needed because H2O imports lm_eval.base,
#    which was removed in lm-eval 0.4.x.
pip install "lm-eval==0.3.0"

# 4. Misc runtime dependency imported by tasks/eval_harness.py.
pip install ftfy
```

## 4. Run Accuracy Evaluation

The accuracy evaluation uses the `h2o_hf` directory and the lm-evaluation-harness pipeline.

### 4.1 Generate Few-Shot Evaluation Prompts

```bash
cd ~/H2O/h2o_hf

python -u generate_task_data.py \
    --output-file openbookqa-5.jsonl \
    --task-name   openbookqa \
    --num-fewshot 5
```

### 4.2 Full Cache Baseline

```bash
time python -u run_lm_eval_harness.py \
    --input-path  openbookqa-5.jsonl \
    --output-path openbookqa-5-opt-full.jsonl \
    --model-name  facebook/opt-6.7b \
    --model-type  opt

python -u evaluate_task_result.py \
    --result-file openbookqa-5-opt-full.jsonl \
    --task-name   openbookqa \
    --num-fewshot 5 \
    --model-type  opt
```

### 4.3 Local Baseline: 20% Recent Only

```bash
time python -u run_lm_eval_harness.py \
    --input-path  openbookqa-5.jsonl \
    --output-path openbookqa-5-opt-local.jsonl \
    --model-name  facebook/opt-6.7b \
    --model-type  opt \
    --enable_small_cache \
    --heavy_ratio 0 \
    --recent_ratio 0.2

python -u evaluate_task_result.py \
    --result-file openbookqa-5-opt-local.jsonl \
    --task-name   openbookqa \
    --num-fewshot 5 \
    --model-type  opt
```

### 4.4 H2O: 10% Heavy + 10% Recent

```bash
time python -u run_lm_eval_harness.py \
    --input-path  openbookqa-5.jsonl \
    --output-path openbookqa-5-opt-h2o.jsonl \
    --model-name  facebook/opt-6.7b \
    --model-type  opt \
    --enable_small_cache \
    --heavy_ratio  0.1 \
    --recent_ratio 0.1

python -u evaluate_task_result.py \
    --result-file openbookqa-5-opt-h2o.jsonl \
    --task-name   openbookqa \
    --num-fewshot 5 \
    --model-type  opt
```

## 5. Run FlexGen System-Performance Experiments

Install FlexGen dependencies:

```bash
cd ~/H2O/h2o_flexgen/
pip install -e .
```

All FlexGen experiments were run from the `h2o_flexgen` directory:

```bash
cd ~/H2O/h2o_flexgen
```

### 5.1 Small Workload: Batch 4, Prompt 512, Generation 512

Full cache:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 4 \
  --num-gpu-batches 1 \
  --prompt-len 512 \
  --gen-len 512 \
  --cut-gen-len 128
```

H2O:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 4 \
  --num-gpu-batches 1 \
  --prompt-len 512 \
  --gen-len 512 \
  --cut-gen-len 128 \
  --hh-ratio 0.2 \
  --hh-all
```

### 5.2 Large Workload: Batch 16, Prompt 2048, Generation 2048

Full cache:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 16 \
  --num-gpu-batches 1 \
  --prompt-len 2048 \
  --gen-len 2048 \
  --cut-gen-len 256
```

H2O:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 16 \
  --num-gpu-batches 1 \
  --prompt-len 2048 \
  --gen-len 2048 \
  --cut-gen-len 256 \
  --hh-ratio 0.2 \
  --hh-all
```

### 5.3 Stress Workload: Batch 24, Prompt 2048, Generation 2048

Full cache:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 24 \
  --num-gpu-batches 1 \
  --prompt-len 2048 \
  --gen-len 2048 \
  --cut-gen-len 256
```

H2O:

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 24 \
  --num-gpu-batches 1 \
  --prompt-len 2048 \
  --gen-len 2048 \
  --cut-gen-len 256 \
  --hh-ratio 0.2 \
  --hh-all
```

## 6. Run Adaptive H2O

For the adaptive H2O extension, the reference length was set as a tunable value:

```bash
REF=32
```

The adaptive configuration uses a square-root scaling rule to reduce the heavy-hitter budget:

```bash
--hh-adaptive sqrt \
--hh-adaptive-ref-len ${REF} \
--hh-min-ratio 0.02
```

### 6.1 Adaptive FlexGen Run: Stress Workload

```bash
python -m flexgen.flex_opt \
  --model facebook/opt-6.7b \
  --path _DUMMY_ \
  --percent 100 0 100 0 100 0 \
  --overlap False \
  --gpu-batch-size 24 \
  --num-gpu-batches 1 \
  --prompt-len 2048 \
  --gen-len 2048 \
  --cut-gen-len 256 \
  --hh-ratio 0.2 \
  --hh-all \
  --hh-adaptive sqrt \
  --hh-adaptive-ref-len ${REF} \
  --hh-min-ratio 0.02
```

### 6.2 Adaptive LM Evaluation: OpenBookQA

The adaptive configuration was also evaluated using the lm-eval-harness pipeline.

Run adaptive H2O:

```bash
python -u run_lm_eval_harness.py \
  --input-path openbookqa-5.jsonl \
  --output-path openbookqa-5-opt-h2o-adaptive.jsonl \
  --model-name facebook/opt-6.7b \
  --model-type opt \
  --enable_small_cache \
  --heavy_ratio 0.1 \
  --recent_ratio 0.1 \
  --hh-adaptive sqrt \
  --hh-adaptive-ref-len ${REF} \
  --hh-min-ratio 0.02
```

Evaluate results:

```bash
python -u evaluate_task_result.py \
  --result-file openbookqa-5-opt-h2o-adaptive.jsonl \
  --task-name openbookqa \
  --num-fewshot 5 \
  --model-type opt
```

This evaluates whether the adaptive cache sizing maintains accuracy while reducing the heavy-hitter budget.
