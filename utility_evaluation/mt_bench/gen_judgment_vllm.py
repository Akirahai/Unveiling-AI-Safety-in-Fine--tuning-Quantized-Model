#!/usr/bin/env python
# coding: utf-8

"""
Generate judgments for MT-bench using vLLM with gpt-oss-20b as judge.
"""

import os
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from utils.read_data import load_questions, load_answers, load_judge_prompts
from utils.VLLMJudge import VLLMJudge
from utils.make_judge import make_judge_prompt_single
from metric import validate_judgment



def judge_single_answer(judge: VLLMJudge, questions: List[Dict], model_answers: Dict,
                       judge_prompts: Dict[str, Dict[str, str]], output_file: str, ref_answers: Optional[Dict] = None,
                       batch_size: int = 8) -> Tuple[List[str], List[str], List[Dict]]:
    """Judge single model answers using vLLM, both single-turn and multi-turn."""
    
    # Categories that need reference answers
    NEED_REF_CATS = ["math", "reasoning", "coding"]
    
    all_judgments = []
    
    # Single turn judgments with both single-v1 and single-math-v1, both single-turn and multi-turn
    print("Processing single-turn judgments...")
    prompts_batch = []
    metadata_batch = []

    # Get model name from one of the answers
    model_name = model_answers[81].get('model_id', 'unknown_model')
    
    # Create Question Metadata
    for question in questions:
        q_id = question["question_id"]
        if q_id not in model_answers:
            continue
            
        answer = model_answers[q_id]
        
        # Choose appropriate judge template
        if question["category"] in NEED_REF_CATS and ref_answers:
            judge_template = judge_prompts["single-math-v1"]
            ref_answer = ref_answers.get(q_id, None)
            if ref_answer is None:
                print(f"Warning: No reference answer for question {q_id}")
                ref_answer = question.get('reference', None)
                if ref_answer is None:
                    print(f"Warning: No reference answer found in question {q_id}")
        else:
            judge_template = judge_prompts["single-v1"]
            ref_answer = None

        system_prompt, user_judge_prompt = make_judge_prompt_single(question, answer, judge_template, ref_answer, multi_turn=False)
        prompts_batch.append((system_prompt, user_judge_prompt))

        metadata_batch.append({
            "question_id": f"{q_id}_single_turn",
            "model": model_name,
            "judge": judge.model_name
        })
    
    # Multi-turn judgments (evaluate conversation flow)
    print("Processing multi-turn conversation judgments...")

    for question in questions:
        if len(question["turns"]) != 2:
            continue
            
        q_id = question["question_id"]
        if q_id not in model_answers:
            continue

        answer = model_answers[q_id]

        # Choose appropriate judge template for conversation evaluation
        if question["category"] in NEED_REF_CATS and ref_answers:
            judge_template = judge_prompts["single-math-v1-multi-turn"]
            ref_answer = ref_answers.get(q_id)
            if ref_answer is None:
                print(f"Warning: No reference answer for question {q_id}")
                ref_answer = question.get('reference', None)
                if ref_answer is None:
                    print(f"Warning: No reference answer found in question {q_id}")
        else:
            judge_template = judge_prompts["single-v1-multi-turn"]
            ref_answer = None
        
        system_prompt, user_judge_prompt = make_judge_prompt_single(question, answer, judge_template, ref_answer, multi_turn=True)
        prompts_batch.append((system_prompt, user_judge_prompt))

        metadata_batch.append({
            "question_id": f"{q_id}_conversation",  # Distinguish from single-turn
            "model": model_name,
            "judge": judge.model_name
        })
    
    # Process in batches
    print(f"Total prompts to judge: {len(prompts_batch)}")
    print(f"Processing in batches of {batch_size}")
    
    for i in tqdm(range(0, len(prompts_batch), batch_size), desc="Judging"):
        batch_prompts = prompts_batch[i:i+batch_size]
        batch_metadata = metadata_batch[i:i+batch_size]
        
        # Separate system and user prompts
        batch_system_prompts = [prompt[0] for prompt in batch_prompts]
        batch_user_prompts = [prompt[1] for prompt in batch_prompts]
        
        # Get judgments for this batch
        judgments = judge.judge_batch(batch_system_prompts, batch_user_prompts)
        
        # Process results
        for (sys_prompt, user_prompt), judgment, metadata in zip(batch_prompts, judgments, batch_metadata):
            result = {
                "question_id": metadata["question_id"],
                "model": metadata["model"],
                "judge": metadata["judge"],
                "judgment": judgment,
                "tstamp": time.time()
            }
            all_judgments.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for judgment in all_judgments:
            f.write(json.dumps(judgment) + "\n")
    
    print(f"Judgments saved to: {output_file}")
    print(f"Total judgments: {len(all_judgments)}")

    return all_judgments
    
    # # Return data needed for evaluation
    # all_system_prompts = [prompt[0] for prompt in prompts_batch]
    # all_user_prompts = [prompt[1] for prompt in prompts_batch]
    # all_questions_metadata = metadata_batch
    
    # return all_system_prompts, all_user_prompts, all_questions_metadata

def main():
    parser = argparse.ArgumentParser(description="Generate MT-bench judgments using vLLM")
    parser.add_argument("--gpus", type=int, nargs='+', default=[0], help="List of GPU ids to use")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-20b", 
                       help="Judge model name")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, 
                       help="Tensor parallel size for judge model")
    parser.add_argument("--answer-dir", type=str, default="data/model_answer",
                       help="Directory containing model answers")
    parser.add_argument("--output-dir", type=str, default="data/model_judgment",
                       help="Directory to save judgments")
    parser.add_argument("--model-dir-list", type=str, default=None,
                       help="List of model directories to judge")
    parser.add_argument("--question-file", type=str, default="data/question.jsonl",
                       help="Questions file")
    parser.add_argument("--judge-file", type=str, default="data/judge_prompts.jsonl",
                       help="Judge prompts file")
    parser.add_argument("--output-file", type=str, default=None,
                       help="Output file for judgments")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--first-n", type=int, default=None,
                       help="Only judge first n questions (for debugging)")
    parser.add_argument("--ref-answers", type=str, default="data/reference_answer/gpt-4.jsonl",
                       help="Reference answers file")
    parser.add_argument("--evaluate", action="store_true",
                       help="Run evaluation after generating judgments")
    parser.add_argument("--evaluation-save-path", type=str, default=None,
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    
    GPU_list = ','.join(map(str, args.gpus))
    
    
    os.environ['CUDA_DEVICE_ORDER'] =  'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES']=  GPU_list
    # Remove WORLD_SIZE to avoid distributed training issues
    # os.environ["WORLD_SIZE"] = "1"  # This was causing the error
    print(f"Using GPU: {GPU_list}")
    
    
    # Load data
    questions = load_questions(args.question_file)
    if args.first_n:
        questions = questions[:args.first_n]
    
    if args.model_dir_list is None:
        raise ValueError("No model answers loaded. Please check the model directories.")
    else:
        print(f"Loaded answers for models: {args.model_dir_list}")
        model_answers = load_answers(os.path.join(args.answer_dir, f"{args.model_dir_list}.jsonl"))

    ref_answers = load_answers(args.ref_answers)
    
    
    judge_prompts = load_judge_prompts(args.judge_file)

    # Set output file
    if args.output_file is None:
        judge_name = args.judge_model.replace("/", "_")
        output_file = os.path.join(args.output_dir, f"{judge_name}/{args.model_dir_list}.jsonl")
    else:
        output_file = os.path.join(args.output_dir, args.output_file)


    print(f"Questions: {len(questions)}")
    print(f"Judge model: {args.judge_model}")
    
    # Initialize judge
    judge = VLLMJudge(args.judge_model, args.tensor_parallel_size)
    
    # Run judgment
    final_judgments = judge_single_answer(
        judge=judge,
        questions=questions,
        model_answers=model_answers,
        judge_prompts=judge_prompts,
        output_file=output_file,
        batch_size=args.batch_size,
        ref_answers= ref_answers
    )
    

    # # Run evaluation if requested
    # if args.evaluate:
    #     print("\n" + "="*50)
    #     print("Running MT-bench evaluation...")
    #     print("="*50)
        
    #     # Set evaluation save path
    #     if args.evaluation_save_path is None:
    #         eval_name = f"data/model_judgment/{judge_name}/{args.model_dir_list}_evaluation.json"
        
    #     # Create sampling params for evaluation (may need different settings)
    #     eval_sampling_params = SamplingParams(
    #         temperature=0.0,
    #         top_p=1.0,
    #         max_tokens=50,  # Limited tokens for rating completion
    #         stop=["]]", "\n"]  # Stop at rating end or newline
    #     )
        
    #     try:
    #         metrics = validate_judgment(
    #             llm=judge.llm,
    #             sampling_params=eval_sampling_params,
    #             system_prompts=system_prompts,
    #             user_prompts=user_prompts,
    #             tokenizer=judge.tokenizer,
    #             questions=questions_metadata,
    #             model_name=args.model_dir_list,
    #             save_path=eval_name
    #         )
            
    #         print(f"\nEvaluation completed successfully!")
    #         print(f"Final Utility Score: {metrics['utility_score']:.3f}")
            
    #     except Exception as e:
    #         print(f"Error during evaluation: {e}")
    #         print("Judgment generation completed successfully, but evaluation failed.")

if __name__ == "__main__":
    main()