#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import warnings
from typing import List
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
import json
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import json

path = 'TheBloke/Llama-2-7B-Chat-GPTQ'
# saved_peft_model_path = "samsumBad-7b-gptq-chat_final"
saved_peft_model_path = "safeLora-samsumBad-7b-gptq-chat_final_0.4"
# saved_peft_model_path = "None"

# Initialize vLLM with LoRA support
tensor_parallel_size = 1  # Adjust based on available GPUs
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic generation for evaluation
    top_p=1.0,
    max_tokens=1024,
    stop=["</s>", "<|im_end|>"]
)

# Load base model with LoRA enabled
llm = LLM(model=path, enable_lora=True, tensor_parallel_size=tensor_parallel_size)

# Set up LoRA request if using adapters
lora_request = None
if saved_peft_model_path.startswith('safeLora'):
    lora_path = f'../finetuned_models/safeLora/{saved_peft_model_path}'
    lora_request = LoRARequest("safe_lora_adapter", 1, lora_path)
    print(f"Using SafeLoRA adapter: {lora_path}")
elif saved_peft_model_path.startswith('samsum'):
    lora_path = f'../finetuned_models/{saved_peft_model_path}'
    lora_request = LoRARequest("samsum_adapter", 1, lora_path)
    print(f"Using SamSum adapter: {lora_path}")
else:
    print("Evaluate the original chat model without LoRA adapters")

tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)



system_msg = "You are a helpful assistant for dialog summarization."

template = {
    "description": "Template used for DialogSummary dataset",
    "prompt": "[INST] <<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_msg} [/INST]",
    "response_split": " [/INST]"
}



from evaluate import load
# Load the ROUGE metric
import evaluate
rouge = evaluate.load('rouge')

def evaluate_batch(prompts, answers):
    """
    Batch evaluation using vLLM for parallel inference with LoRA support
    """
    # Generate responses for all prompts in parallel with LoRA request if available
    if lora_request is not None:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sampling_params)
    
    results = []
    for i, output in enumerate(outputs):
        prediction = output.outputs[0].text.strip()
        
        # # Clean up prediction
        # if prediction.startswith("Sure,"):
        #     pattern = re.compile(r'^Sure,.*')
        #     lines = prediction.split('\n')
        #     filtered_lines = [line for line in lines if not pattern.match(line)]
        #     prediction_formatted = '\n'.join(filtered_lines)
        #     print("Filtered 'Sure,*' line from prediction.")
        # else:
        prediction_formatted = prediction

        # print(f"Prediction {i}:", prediction_formatted)
        # print(f"Answer {i}:", answers[i])
        
        # Calculate ROUGE score
        rouge_result = rouge.compute(predictions=[prediction_formatted], references=[answers[i]])
        results.append(rouge_result['rouge1'])
    
    return results



# Batch processing for parallel inference
batch_size = 64 # Adjust based on GPU memory
prompts = []
answers = []
all_f1_scores = []

# Load all data first
with open('datasets/samsum_test.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if line.strip() and i < 200:  # check if line is not empty
            question = json.loads(line)["messages"]
            
            # Format prompt
            if 'llama-3' in path or 'gemma' in path:
                prompt = system_msg + question[0]["content"]
            else:
                prompt = template["prompt"].format(system_msg=system_msg, user_msg=question[0]['content'])
            
            prompts.append(prompt)
            answers.append(question[1]['content'])

# Process in batches for parallel inference
print(f"Processing {len(prompts)} prompts in batches of {batch_size}")
for i in range(0, len(prompts), batch_size):
    batch_prompts = prompts[i:i+batch_size]
    batch_answers = answers[i:i+batch_size]
    
    print(f"===== Processing batch {i//batch_size + 1} (samples {i}-{min(i+batch_size-1, len(prompts)-1)}) ==========")
    
    # Evaluate batch
    batch_f1_scores = evaluate_batch(batch_prompts, batch_answers)
    all_f1_scores.extend(batch_f1_scores)

# Calculate average F1 score
average_f1 = sum(all_f1_scores) / len(all_f1_scores)
print(f'Average Rouge F1 Score: {average_f1}')

# Store all evaluation results in a txt file
# Create results directory if it doesn't exist
model_path = path.split('/')[-1]
os.makedirs(f'results/{model_path}', exist_ok=True)
with open(f'results/{model_path}/eval_results_{model_path}_{saved_peft_model_path}.txt', 'w') as f:
    f.write(f'Evaluation Results for model: {model_path}_{saved_peft_model_path} on samsum test dataset\n')
    f.write(f'Base model: {path}\n')
    if lora_request is not None:
        f.write(f'LoRA adapter: {lora_request.lora_local_path}\n')
        f.write(f'LoRA adapter name: {lora_request.lora_name}\n')
    else:
        f.write('LoRA adapter: None (using base model only)\n')
    f.write(f'Average Rouge F1 Score: {average_f1}\n')
    f.write(f'Total samples evaluated: {len(all_f1_scores)}\n')
    f.write(f'Batch size used: {batch_size}\n')