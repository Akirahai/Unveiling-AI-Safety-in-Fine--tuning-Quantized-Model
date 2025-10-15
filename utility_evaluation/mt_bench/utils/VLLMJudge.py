import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


class VLLMJudge:
    """Judge using vLLM for inference."""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 2):
        """Initialize the judge with vLLM."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Set up stop tokens
        stop_tokens = self.tokenizer.additional_special_tokens
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size
        )
        
        self.sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for consistent judging
            top_p=1.0,
            max_tokens=1024,
            stop=stop_tokens
        )
        
        print(f"Initialized VLLMJudge with model: {model_name}")
    
    def judge_single(self, system_prompt: str, user_prompt: str) -> str:
        """Judge a single prompt using chat template with system prompt."""
        # Use chat template to properly format system and user messages
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.generate([formatted_prompt], self.sampling_params)
        result = outputs[0].outputs[0].text.strip()
        return result
    
    def judge_batch(self, system_prompts: List[str], user_prompts: List[str]) -> List[str]:
        """Judge a batch of prompts using chat template with system prompts."""
        # Format all conversations with chat template
        formatted_prompts = []
        for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
            conversation = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        
        initial_completions = self.llm.generate(formatted_prompts, self.sampling_params)
        initial_texts = [comp.outputs[0].text.strip() for comp in initial_completions]

        return initial_texts
