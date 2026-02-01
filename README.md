# vLLM Evaluation Framework

Run stability evaluations on open-source LLMs using vLLM inference servers.

## Install (uv)

```bash
uv sync
```

## Quick start (dev mode)

```bash
python scripts/run_pipeline.py \
  --models qwq-32b gpt-oss-20b \
  --benchmarks mmlu mt_bench multichallenge \
  --runs 1 \
  --temperature 0.7 \
  --judge-model gpt-oss-120b \
  --judge-votes 3 \
  --output-dir ./outputs \
  --dev-mode
```

## Inference only

```bash
python scripts/run_inference.py \
  --models qwq-32b gpt-oss-20b \
  --benchmarks mmlu mt_bench multichallenge \
  --runs 5 \
  --temperature 0.7 \
  --simultaneous \
  --batch-size 32 \
  --output-dir ./outputs/inference \
  --dev-mode \
  --resume
```

## Evaluation only

```bash
python scripts/run_evaluation.py \
  --inference-dir ./outputs/inference \
  --benchmarks mmlu mt_bench multichallenge \
  --judge-model gpt-oss-120b \
  --judge-votes 3 \
  --output-dir ./outputs \
  --resume
```

## Layout

- `configs/`: model + benchmark configs
- `scripts/`: CLI entrypoints
- `src/`: framework code
- `outputs/`: inference, judgments, checkpoints, final report
