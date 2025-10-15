#!/usr/bin/env python
# coding: utf-8

"""
Test script for MT-bench evaluation metrics.
"""

import sys
import os
sys.path.append('/home/long_2/hai/Individual_Project/Safe_Quantized_LoRA/utility_evaluation/mt_bench')

from metric import load_and_evaluate_judgments

def test_evaluation():
    """Test the evaluation function with existing judgment file."""
    
    # Path to existing judgment file
    judgment_file = "/home/long_2/hai/Individual_Project/Safe_Quantized_LoRA/utility_evaluation/mt_bench/data/model_judgment/openai_gpt-oss-20b_single.jsonl"
    
    if not os.path.exists(judgment_file):
        print(f"Judgment file not found: {judgment_file}")
        return
    
    print("Testing evaluation function...")
    
    # Set save path for results
    save_path = "test_results/evaluation_test.json"
    
    try:
        metrics = load_and_evaluate_judgments(judgment_file, save_path)
        
        print("\n=== Test Results ===")
        print(f"Evaluation completed successfully!")
        print(f"Utility Score: {metrics['utility_score']:.3f}")
        print(f"Average Rating: {metrics['average_rating']:.2f}/10")
        print(f"Total Questions: {metrics['total_questions']}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluation()