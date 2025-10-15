
model_name="llama2-7b-gptq-base"
# judge_model="openai/gpt-oss-20b"
judge_model="Qwen/Qwen2.5-3B-Instruct"


python gen_judgment_vllm.py \
    --judge-model $judge_model \
    --tensor-parallel-size 2 \
    --model-dir-list $model_name \
    --batch-size 4 \
    --gpus 2 3 \
    --first-n 5