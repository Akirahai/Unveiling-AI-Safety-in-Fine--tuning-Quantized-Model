# MT-Bench Evaluation with vLLM

This directory contains scripts for running MT-bench evaluation using vLLM for both answer generation and judgment.

## Features

- **Fast Answer Generation**: Uses vLLM for parallel inference with LoRA support
- **Automated Judgment**: Uses gpt-oss-20b as judge model via vLLM
- **Batch Processing**: Efficient batch inference for both generation and judgment
- **LoRA Support**: Can evaluate models with LoRA adapters
- **Complete Pipeline**: Single script to run end-to-end evaluation

## Files

- `gen_model_answer_vllm.py`: Generate answers using vLLM with LoRA support
- `gen_judgment_vllm.py`: Generate judgments using vLLM with gpt-oss-20b
- `run_mt_bench_vllm.py`: Complete evaluation pipeline
- `README_vllm.md`: This file

## Quick Start

### 1. Complete Pipeline (Recommended)

Run the complete evaluation pipeline:

```bash
# Evaluate base model
python run_mt_bench_vllm.py --model-name "TheBloke/Llama-2-7B-Chat-GPTQ"

# Evaluate model with LoRA adapter
python run_mt_bench_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --peft-model "safeLora-samsumBad-7b-gptq-chat_final_0.4"

# With custom configuration
python run_mt_bench_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --peft-model "safeLora-samsumBad-7b-gptq-chat_final_0.4" \
    --model-id "llama2-7b-safeLora" \
    --tensor-parallel-size 2 \
    --batch-size 16 \
    --judge-model "openai/gpt-oss-20b"
```

### 2. Step-by-Step Execution

#### Generate Answers

```bash
# Base model
python gen_model_answer_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --model-id "llama2-7b-base" \
    --tensor-parallel-size 1 \
    --batch-size 8

# With LoRA adapter
python gen_model_answer_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --peft-model "safeLora-samsumBad-7b-gptq-chat_final_0.4" \
    --model-id "llama2-7b-safeLora" \
    --tensor-parallel-size 1 \
    --batch-size 8
```

#### Generate Judgments

```bash
python gen_judgment_vllm.py \
    --judge-model "openai/gpt-oss-20b" \
    --tensor-parallel-size 2 \
    --batch-size 8 \
    --model-list "llama2-7b-base" "llama2-7b-safeLora"
```

## Configuration Options

### Model Configuration
- `--model-name`: Base model name/path (required)
- `--model-id`: Custom identifier for output files
- `--peft-model`: LoRA adapter path (optional)

### Hardware Configuration
- `--tensor-parallel-size`: Number of GPUs for model parallelism
- `--judge-tensor-parallel-size`: Number of GPUs for judge model
- `--batch-size`: Batch size for inference

### Generation Configuration
- `--max-new-tokens`: Maximum tokens to generate (default: 1024)
- `--judge-model`: Judge model name (default: "openai/gpt-oss-20b")

### Pipeline Control
- `--skip-generation`: Skip answer generation step
- `--skip-judgment`: Skip judgment step

## Example Usage Scenarios

### Scenario 1: Compare Base Model vs LoRA

```bash
# Generate answers for base model
python gen_model_answer_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --model-id "llama2-base"

# Generate answers for LoRA model
python gen_model_answer_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --peft-model "safeLora-samsumBad-7b-gptq-chat_final_0.4" \
    --model-id "llama2-safeLora"

# Judge both models
python gen_judgment_vllm.py \
    --model-list "llama2-base" "llama2-safeLora"
```

### Scenario 2: Multi-GPU Setup

```bash
python run_mt_bench_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --tensor-parallel-size 2 \
    --judge-tensor-parallel-size 2 \
    --batch-size 16
```

### Scenario 3: Debug Mode (First 10 Questions)

```bash
python gen_judgment_vllm.py \
    --judge-model "openai/gpt-oss-20b" \
    --model-list "your-model-id" \
    --first-n 10
```

## Output Files

### Answer Files
- Location: `data/model_answer/{model_id}.jsonl`
- Format: Each line contains a JSON object with question_id, answer_id, model_id, choices, and timestamp

### Judgment Files  
- Location: `data/model_judgment/{judge_model}_single.jsonl`
- Format: Each line contains judgment results with scores and reasoning

## Performance Tips

1. **GPU Memory**: Adjust `batch_size` based on available GPU memory
2. **Multi-GPU**: Use `tensor_parallel_size > 1` for large models
3. **LoRA**: LoRA adapters have minimal memory overhead
4. **Judge Model**: gpt-oss-20b requires ~40GB VRAM, use tensor parallelism if needed

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `batch_size` or increase `tensor_parallel_size`
2. **LoRA Not Found**: Check the adapter path relative to script location
3. **Judge Model Issues**: Ensure gpt-oss-20b is available and compatible

### Debugging

```bash
# Test with a single question first
python gen_judgment_vllm.py --first-n 1 --model-list "your-model"

# Check generated answers
cat data/model_answer/your-model.jsonl | head -1 | jq .

# Check judgments
cat data/model_judgment/openai_gpt-oss-20b_single.jsonl | head -1 | jq .
```

## Integration with Your Workflow

Based on your existing SamSum evaluation, you can integrate MT-bench as follows:

```bash
# 1. Evaluate on SamSum (your existing script)
python ../SamSum.py

# 2. Evaluate on MT-bench
python run_mt_bench_vllm.py \
    --model-name "TheBloke/Llama-2-7B-Chat-GPTQ" \
    --peft-model "safeLora-samsumBad-7b-gptq-chat_final_0.4"

# 3. Compare results across different evaluation metrics
```

This gives you comprehensive evaluation across both task-specific (SamSum) and general capabilities (MT-bench) metrics.