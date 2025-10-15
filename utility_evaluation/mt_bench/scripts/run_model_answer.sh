model_name="TheBloke/Llama-2-7B-Chat-GPTQ"
model_id="safeLora-alpaca-llama2-7b-gptq"
peft_model="safeLora-alpaca-7b-gptq-chat_final"

cmd="python gen_model_answer_vllm.py \
    --model-name \"$model_name\" \
    --model-id \"$model_id\" \
    --tensor-parallel-size 1 \
    --batch-size 8 \
    --max-new-tokens 1024 \
    --gpus 5"

# Add --peft-model only if it's not empty
if [ -n "$peft_model" ]; then
    cmd="$cmd --peft-model \"$peft_model\""
fi

eval $cmd