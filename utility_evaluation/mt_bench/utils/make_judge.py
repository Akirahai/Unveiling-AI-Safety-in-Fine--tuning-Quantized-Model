import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm


def make_judge_prompt_single(question: Dict, answer: Dict, judge_template: Dict, 
                           ref_answer: Optional[Dict] = None, multi_turn: bool = False) -> tuple[str, str]:
    """Create judgment prompt for single model answer evaluation, both for multi-turn and single-turn.
    
    Returns:
        tuple: (system_prompt, user_prompt)
    """
    
    # Get system prompt and template
    system_prompt = judge_template.get('system_prompt', '')
    template = judge_template['prompt_template']
    
    if multi_turn:
        user_prompt = question["turns"][0]
        assistant_answer = answer["choices"][0]["turns"][0]
        user_prompt_2 = question["turns"][1]
        assistant_answer_2 = answer["choices"][0]["turns"][1]
        
        if ref_answer:
            ref_answer_1 = ref_answer["choices"][0]["turns"][0]
            ref_answer_2 = ref_answer["choices"][0]["turns"][1]
            user_content = template.format(
                question_1=user_prompt,
                answer_1=assistant_answer,
                question_2=user_prompt_2,
                answer_2=assistant_answer_2,
                ref_answer_1=ref_answer_1,
                ref_answer_2=ref_answer_2
            )
        else:
            user_content = template.format(
                question_1=user_prompt,
                answer_1=assistant_answer,
                question_2=user_prompt_2,
                answer_2=assistant_answer_2
            )
    else:
        user_prompt = question["turns"][0]
        assistant_answer = answer["choices"][0]["turns"][0]
        
        if ref_answer:
            ref_answer_text = ref_answer["choices"][0]["turns"][0]
            user_content = template.format(
                question=user_prompt,
                answer=assistant_answer,
                ref_answer=ref_answer_text
            )
        else:
            user_content = template.format(
                question=user_prompt,
                answer=assistant_answer
            )
    
    return system_prompt, user_content

def make_judge_prompt_pairwise(question: Dict, answer_a: Dict, answer_b: Dict, 
                             judge_template: Dict, ref_answer: Optional[Dict] = None, 
                             multi_turn: bool = False) -> str:
    """Create judgment prompt for pairwise comparison."""
    
    # Get system prompt and template
    system_prompt = judge_template.get('system_prompt', '')
    template = judge_template['prompt_template']
    
    if multi_turn:
        user_prompt = question["turns"][0]
        answer_a_1 = answer_a["choices"][0]["turns"][0]
        answer_b_1 = answer_b["choices"][0]["turns"][0]
        user_prompt_2 = question["turns"][1]
        answer_a_2 = answer_a["choices"][0]["turns"][1]
        answer_b_2 = answer_b["choices"][0]["turns"][1]
        
        if ref_answer:
            ref_answer_1 = ref_answer["choices"][0]["turns"][0]
            ref_answer_2 = ref_answer["choices"][0]["turns"][1]
            user_content = template.format(
                question_1=user_prompt,
                answer_a_1=answer_a_1,
                answer_b_1=answer_b_1,
                question_2=user_prompt_2,
                answer_a_2=answer_a_2,
                answer_b_2=answer_b_2,
                ref_answer_1=ref_answer_1,
                ref_answer_2=ref_answer_2
            )
        else:
            user_content = template.format(
                question_1=user_prompt,
                answer_a_1=answer_a_1,
                answer_b_1=answer_b_1,
                question_2=user_prompt_2,
                answer_a_2=answer_a_2,
                answer_b_2=answer_b_2
            )
    else:
        user_prompt = question["turns"][0]
        answer_a_text = answer_a["choices"][0]["turns"][0]
        answer_b_text = answer_b["choices"][0]["turns"][0]
        
        if ref_answer:
            ref_answer_text = ref_answer["choices"][0]["turns"][0]
            user_content = template.format(
                question=user_prompt,
                answer_a=answer_a_text,
                answer_b=answer_b_text,
                ref_answer=ref_answer_text
            )
        else:
            user_content = template.format(
                question=user_prompt,
                answer_a=answer_a_text,
                answer_b=answer_b_text
            )
    
    # Combine system prompt and user content in chat format
    if system_prompt:
        prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_content}<|end|>\n<|assistant|>\n"
    else:
        prompt = f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n"
    
    return prompt