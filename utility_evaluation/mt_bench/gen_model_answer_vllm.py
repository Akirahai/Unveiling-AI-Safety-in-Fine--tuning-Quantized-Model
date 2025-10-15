#!/usr/bin/env python
# coding: utf-8

"""
Generate answers for the mt-bench 80 questions using vLLM with LoRA support.
"""

import os
import json
import time
# import shortuuid
import argparse
from typing import Optional, List
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

def load_questions(question_file: str, begin: Optional[int] = None, end: Optional[int] = None):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions

# Sampling temperature configs for different categories
temperature_config = {
    "writing": 0.7,
    "roleplay": 0.7,
    "extraction": 0.0,
    "math": 0.0,
    "coding": 0.0,
    "reasoning": 0.0,
    "stem": 0.1,
    "humanities": 0.1,
}

def generate_answers_vllm(
    model_name: str,
    model_id: str = None,
    peft_model: str = None,
    max_new_tokens: int = 1024,
    prompt_file: str = 'data/question.jsonl',
    output_file: str = None,
    tensor_parallel_size: int = 1,
    batch_size: int = 8,
    **kwargs
):
    """
    Generate answers using vLLM with optional LoRA adapters
    """
    
    if model_id is None:
        model_id = model_name.split("/")[-1]
    if output_file is None:
        output_file = f"data/model_answer/{model_id}.jsonl"
    

    print(f"Model: {model_name}")
    print(f"Model ID: {model_id}")
    print(f"PEFT Model: {peft_model if peft_model else 'No LoRA'}")
    print(f"Max New Tokens: {max_new_tokens}")
    # Initialize vLLM with LoRA support
    llm = LLM(
        model=model_name, 
        enable_lora=True if peft_model else False, 
        tensor_parallel_size=tensor_parallel_size
    )
    
    # Set up LoRA request if using adapters
    lora_request = None
    if peft_model:
        if peft_model.startswith('safeLora'):
            lora_path = f'../../finetuned_models/safeLora/{peft_model}'
            lora_request = LoRARequest("safe_lora_adapter", 1, lora_path)
            print(f"Using SafeLoRA adapter: {lora_path}")
        else:
            lora_path = f'../../finetuned_models/{peft_model}'
            lora_request = LoRARequest("samsum_adapter", 1, lora_path)
            print(f"Using SamSum adapter: {lora_path}")

    else:
        print("Using base model without LoRA adapters")

    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    
    # Load questions
    questions = load_questions(prompt_file)
    
    # Template for Llama-2 chat format
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    
    def format_prompt(question: str, system_msg: str = "You are a helpful assistant.") -> str:
        """Format prompt in Llama-2 chat format"""
        return f"{B_INST} {B_SYS}{system_msg}{E_SYS}{question} {E_INST}"
    
    results = []
    
    # Process questions in batches for both turns
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        
        print(f"Processing batch {i//batch_size + 1} (questions {i}-{min(i+batch_size-1, len(questions)-1)})")
        
        # Prepare first turn prompts
        first_turn_prompts = []
        sampling_params_list = []
        
        for question in batch_questions:
            # Configure sampling parameters based on category
            temperature = temperature_config.get(question["category"], 0.7)
            do_sample = temperature >= 1e-4
            
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=1.0,
                max_tokens=max_new_tokens,
                stop=["</s>", "<|im_end|>"]
            )
            
            prompt = format_prompt(question["turns"][0])
            first_turn_prompts.append(prompt)
            sampling_params_list.append(sampling_params)
        
        # Generate first turn responses
        if lora_request:
            first_turn_outputs = llm.generate(first_turn_prompts, sampling_params_list[0], lora_request=lora_request)
        else:
            first_turn_outputs = llm.generate(first_turn_prompts, sampling_params_list[0])
        
        # Prepare second turn prompts
        second_turn_prompts = []
        
        for j, (question, first_output) in enumerate(zip(batch_questions, first_turn_outputs)):
            first_response = first_output.outputs[0].text.strip()
            
            # Create conversation history for second turn
            dialog = f"{first_turn_prompts[j]} {first_response} {B_INST} {question['turns'][1]} {E_INST}"
            second_turn_prompts.append(dialog)
        
        # Generate second turn responses
        if lora_request:
            second_turn_outputs = llm.generate(second_turn_prompts, sampling_params_list[0], lora_request=lora_request)
        else:
            second_turn_outputs = llm.generate(second_turn_prompts, sampling_params_list[0])
        
        # Process results
        for j, (question, first_output, second_output) in enumerate(zip(batch_questions, first_turn_outputs, second_turn_outputs)):
            first_response = first_output.outputs[0].text.strip()
            second_response = second_output.outputs[0].text.strip()
            
            # print(f"\nQuestion {question['question_id']}:")
            # print(f"Turn 1: {question['turns'][0]}")
            # print(f"Answer 1: {first_response}")
            # print(f"Turn 2: {question['turns'][1]}")
            # print(f"Answer 2: {second_response}")
            # print("-" * 50)
            
            result = {
                "question_id": question["question_id"],
                "model_id": model_id,
                "choices": [{
                    "index": 0,
                    "turns": [first_response, second_response]
                }],
                "tstamp": time.time(),
            }
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Total questions processed: {len(results)}")

def main():
    parser = argparse.ArgumentParser(description="Generate MT-bench answers using vLLM")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU ids to use")
    parser.add_argument("--model-name", type=str, required=True, help="Base model name/path")
    parser.add_argument("--model-id", type=str, default=None, help="Model identifier for output")
    parser.add_argument("--peft-model", type=str, default=None, help="LoRA adapter path")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum new tokens to generate")
    parser.add_argument("--prompt-file", type=str, default="data/question.jsonl", help="Questions file")
    parser.add_argument("--output-file", type=str, default=None, help="Output file for answers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference")
    
    args = parser.parse_args()
    
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    # Remove WORLD_SIZE to avoid distributed training issues
    # os.environ["WORLD_SIZE"] = "1"  # This was causing the error
    print(f"Using GPU: {GPU_list}")
    generate_answers_vllm(
        model_name=args.model_name,
        model_id=args.model_id,
        peft_model=args.peft_model,
        max_new_tokens=args.max_new_tokens,
        prompt_file=args.prompt_file,
        output_file=args.output_file,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()