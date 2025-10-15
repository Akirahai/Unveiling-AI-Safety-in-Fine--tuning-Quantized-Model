import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
from tqdm import tqdm


def load_questions(question_file: str) -> List[Dict]:
    """Load questions from file."""
    questions = []
    with open(question_file, "r") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions

# def load_model_answers(answer_dir: str) -> Dict[str, Dict]:
#     """Load model answers from directory."""
#     model_answers = {}
#     print(f"Loading model answers from directory: {answer_dir}")
#     print(os.listdir(answer_dir))
#     for file in os.listdir(answer_dir):
#         if file.endswith('.jsonl'):
#             model_name = file[:-6]  # Remove .jsonl
#             print(f"Loading answers for model: {model_name}")
#             model_answers[model_name] = {}
#             with open(os.path.join(answer_dir, file), 'r') as f:
#                 for line in f:
#                     if line.strip():
#                         answer = json.loads(line)
#                         model_answers[model_name][answer['question_id']] = answer
#     return model_answers

def load_answers(answer_file: str) -> Dict[str, Dict]:
    """Load answers from a single file."""
    model_answers = {}
    with open(answer_file, 'r') as f:
        for line in f:
            if line.strip():
                answer = json.loads(line)
                model_answers[answer['question_id']] = answer
    return model_answers


def load_judge_prompts(judge_file: str) -> Dict[str, Dict[str, str]]:
    """Load judge prompts from file."""
    judge_prompts = {}
    with open(judge_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                judge_prompts[data['name']] = {
                    'system_prompt': data.get('system_prompt', ''),
                    'prompt_template': data['prompt_template']
                }
    return judge_prompts