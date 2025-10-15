#!/usr/bin/env python
# coding: utf-8

"""
Example usage of MT-bench judgment generation with evaluation.

This script demonstrates how to:
1. Generate judgments using vLLM
2. Automatically evaluate and extract ratings
3. Compute utility scores and metrics
"""

import os

def run_mt_bench_with_evaluation():
    """Run MT-bench judgment generation with evaluation."""
    
    # Example command for running MT-bench with evaluation
    cmd_parts = [
        "python gen_judgment_vllm.py",
        "--gpus 0 1",
        "--judge-model openai/gpt-oss-20b",
        "--tensor-parallel-size 2", 
        "--answer-dir data/model_answer",
        "--model-dir-list safeLora-samsumBad-7b-gptq-chat_final",  # Your model name
        "--question-file data/question.jsonl",
        "--judge-file data/judge_prompts.jsonl",
        "--ref-answers data/reference_answer/gpt-4.jsonl",
        "--batch-size 4",
        "--first-n 10",  # Test with first 10 questions
        "--evaluate",  # Enable evaluation
        "--evaluation-save-path results/safeLora_evaluation.json"
    ]
    
    command = " ".join(cmd_parts)
    print("Command to run:")
    print(command)
    print("\n" + "="*80)
    
    # You can uncomment the line below to actually run it
    # os.system(command)

def evaluate_existing_judgments():
    """Evaluate existing judgment files."""
    
    # Example of evaluating an existing judgment file
    cmd_parts = [
        "python metric.py",
        "--judgment-file data/model_judgment/openai_gpt-oss-20b_single.jsonl",
        "--save-path results/existing_evaluation.json"
    ]
    
    command = " ".join(cmd_parts)
    print("Command to evaluate existing judgments:")
    print(command)
    print("\n" + "="*80)
    
    # You can uncomment the line below to actually run it
    # os.system(command)

if __name__ == "__main__":
    print("=== MT-bench with Evaluation Examples ===\n")
    
    print("1. Generate judgments with automatic evaluation:")
    run_mt_bench_with_evaluation()
    
    print("\n2. Evaluate existing judgment file:")
    evaluate_existing_judgments()
    
    print("\n=== Key Features ===")
    print("✓ Continues generation if rating format [[*]] not found")
    print("✓ Extracts ratings from completion text")
    print("✓ Computes utility scores and detailed metrics")
    print("✓ Saves results to CSV and JSON formats")
    print("✓ Provides category-wise breakdowns")
    print("✓ Handles both single-turn and multi-turn evaluations")
    
    print("\n=== Output Files ===")
    print("- Judgments: data/model_judgment/{judge_model}/{model_name}.jsonl")
    print("- Evaluation CSV: results/{model_name}_evaluation.csv")
    print("- Metrics JSON: results/{model_name}_evaluation.json")