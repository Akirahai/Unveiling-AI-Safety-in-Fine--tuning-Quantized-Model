#!/bin/bash

# Define the models and datasets
models=(
  "TheBloke/Llama-2-7B-Chat-fp16"
  # "TheBloke/Llama-2-7B-Chat-GPTQ"
)


saved_models=(
  # "samsum-7b-gptq-chat"
  "samsum-7b-fp16-chat"
  # "samsumBad-7b-gptq-chat"
  # "samsumBad-7b-fp16-chat"
  # "pureBad-7b-gptq-chat"
  # "pureBad-7b-fp16-chat"
  # "alpaca-7b-gptq-chat"
  # "alpaca-7b-fp16-chat"
)

datasets=("samsum")


# Hyperparameters for finetuning PureBad and DialogSummary
lrs=(5e-5)
batch_sizes=(5)
num_epochs=(5)

# Hyperparameters for finetuning Alpaca
# lrs=(2e-5)
# batch_sizes=(32)
# num_epochs=(1)

# Loop through models
for i in "${!models[@]}"; do
  model=${models[$i]}
  # aligned_model=${aligned_models[$i]}
  saved_model=${saved_models[$i]}
  lr=${lrs[$i]}
  bs=${batch_sizes[$i]}
  epochs=${num_epochs[$i]}

  for dataset in "${datasets[@]}"; do
    echo "Launching fine-tuning on GPU $gpu: $model"
    python finetune_model.py \
      --gpus 1 --use_gpu \
      --data_path "$dataset" \
      --model "$model" \
      --saved_peft_model "$saved_model" \
      --lr "$lr" \
      --batch_size "$bs" \
      --num_epochs "$epochs"
  done
done