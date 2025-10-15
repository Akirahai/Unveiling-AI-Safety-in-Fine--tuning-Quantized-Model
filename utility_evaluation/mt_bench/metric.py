#!/usr/bin/env python
# coding: utf-8

"""
Metrics computation for MT-bench judgments.
"""

import os
import re
import json
import csv
import torch
from typing import List, Dict, Any, Optional, Tuple
from vllm import SamplingParams
import numpy as np


@torch.no_grad()
def validate_judgment(llm, sampling_params, system_prompts: List[str], user_prompts: List[str], 
                     tokenizer, questions: List[Dict], model_name: str = None, 
                     save_path: str = None) -> Dict[str, float]:
    """
    Validate MT-bench judgments by ensuring proper rating format and extracting scores.
    
    Args:
        llm: vLLM model instance
        sampling_params: Sampling parameters for generation
        system_prompts: List of system prompts for each judgment
        user_prompts: List of user prompts for each judgment
        tokenizer: Tokenizer for chat template formatting
        questions: List of question metadata
        model_name: Name of the model being evaluated
        save_path: Path to save detailed results
    
    Returns:
        Dictionary with computed metrics
    """
    
    # Format prompts using chat template
    formatted_prompts = []
    for sys_prompt, user_prompt in zip(system_prompts, user_prompts):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)
    
    # Initial generation
    print("Generating initial judgments...")
    initial_completions = llm.generate(formatted_prompts, sampling_params=sampling_params)
    initial_texts = [comp.outputs[0].text for comp in initial_completions]
    
    # Check which completions need continuation (don't have proper rating format)
    rating_pattern = r'\n\nRating: \[\[(\d+)\]\]'
    final_completions = [None] * len(formatted_prompts)
    continuation_prompts = []
    continuation_indices = []
    initial_completions_to_continue = []
    
    print("Checking for proper rating format...")
    for idx, completion in enumerate(initial_texts):
        if re.search(rating_pattern, completion):
            # Found proper rating format
            final_completions[idx] = completion
        else:
            # Need to continue generation with rating prompt
            continuation_prompt = formatted_prompts[idx] + completion + "\n\nRating: [["
            continuation_prompts.append(continuation_prompt)
            continuation_indices.append(idx)
            initial_completions_to_continue.append(completion + "\n\nRating: [[")
    
    # Generate continuations if needed
    if len(continuation_prompts) > 0:
        print(f"Continuing generation for {len(continuation_prompts)} incomplete judgments...")
        continuation_completions = llm.generate(continuation_prompts, sampling_params=sampling_params)
        continuation_texts = [comp.outputs[0].text for comp in continuation_completions]
        
        # Combine initial and continuation
        for idx, cont_idx, initial_part, continuation in zip(
            range(len(continuation_indices)), continuation_indices, 
            initial_completions_to_continue, continuation_texts
        ):
            final_completions[cont_idx] = initial_part + continuation
    
    # Extract ratings and compute metrics
    print("Extracting ratings and computing metrics...")
    results = extract_ratings_and_compute_metrics(
        final_completions, questions, model_name, save_path
    )
    
    return results


def extract_ratings_and_compute_metrics(completions: List[str], questions: List[Dict], 
                                      model_name: str = None, save_path: str = None) -> Dict[str, float]:
    """
    Extract rating scores from completions and compute utility metrics.
    
    Args:
        completions: List of judgment completions
        questions: List of question metadata
        model_name: Name of the model being evaluated
        save_path: Path to save detailed results
    
    Returns:
        Dictionary with computed metrics
    """
    
    # Pattern to extract rating from completion
    rating_pattern = r'Rating: \[\[(\d+)\]\]'
    
    extracted_ratings = []
    detailed_results = []
    
    for idx, completion in enumerate(completions):
        question = questions[idx] if idx < len(questions) else {}
        
        # Extract rating from the completion part only
        matches = re.findall(rating_pattern, completion)
        
        if matches:
            # Take the last match (most recent rating)
            rating = int(matches[-1])
        else:
            # No rating found, assign default low score
            rating = 1
            print(f"Warning: No rating found in completion {idx}, assigning rating 1")
        
        extracted_ratings.append(rating)
        
        # Store detailed results
        result_entry = {
            'question_id': question.get('question_id', idx),
            'category': question.get('category', 'unknown'),
            'rating': rating,
            'completion': completion,
            'model': model_name or 'unknown'
        }
        detailed_results.append(result_entry)
    
    # Compute metrics
    ratings_array = np.array(extracted_ratings)
    
    metrics = {
        'total_questions': len(extracted_ratings),
        'average_rating': float(np.mean(ratings_array)),
        'median_rating': float(np.median(ratings_array)),
        'std_rating': float(np.std(ratings_array)),
        'min_rating': float(np.min(ratings_array)),
        'max_rating': float(np.max(ratings_array)),
        'ratings_distribution': {
            str(i): int(np.sum(ratings_array == i)) for i in range(1, 11)
        }
    }
    
    # Calculate utility score (average rating normalized to 0-1 scale)
    utility_score = metrics['average_rating'] / 10.0
    metrics['utility_score'] = utility_score
    
    # Compute category-wise metrics
    category_metrics = {}
    for result in detailed_results:
        category = result['category']
        if category not in category_metrics:
            category_metrics[category] = []
        category_metrics[category].append(result['rating'])
    
    for category, ratings in category_metrics.items():
        cat_array = np.array(ratings)
        metrics[f'{category}_avg_rating'] = float(np.mean(cat_array))
        metrics[f'{category}_utility_score'] = float(np.mean(cat_array)) / 10.0
        metrics[f'{category}_count'] = len(ratings)
    
    # Save detailed results if path provided
    if save_path:
        save_detailed_results(detailed_results, metrics, save_path)
    
    # Print summary
    print(f"\n=== MT-bench Evaluation Results ===")
    print(f"Model: {model_name}")
    print(f"Total Questions: {metrics['total_questions']}")
    print(f"Average Rating: {metrics['average_rating']:.2f}/10")
    print(f"Utility Score: {metrics['utility_score']:.3f}")
    print(f"Standard Deviation: {metrics['std_rating']:.2f}")
    print(f"Rating Range: {metrics['min_rating']:.0f} - {metrics['max_rating']:.0f}")
    
    print(f"\n=== Category Breakdown ===")
    for category in category_metrics.keys():
        avg_rating = metrics[f'{category}_avg_rating']
        utility = metrics[f'{category}_utility_score']
        count = metrics[f'{category}_count']
        print(f"{category}: {avg_rating:.2f}/10 (utility: {utility:.3f}, n={count})")
    
    return metrics


def save_detailed_results(detailed_results: List[Dict], metrics: Dict, save_path: str):
    """Save detailed results to CSV and JSON files."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save detailed results as CSV
    csv_path = save_path.replace('.json', '.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        if detailed_results:
            fieldnames = detailed_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detailed_results)
    
    # Save metrics as JSON
    json_path = save_path.replace('.csv', '.json')
    with open(json_path, 'w', encoding='utf-8') as jsonfile:
        json.dump(metrics, jsonfile, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved to: {csv_path}")
    print(f"Metrics summary saved to: {json_path}")


def load_and_evaluate_judgments(judgment_file: str, save_path: str = None) -> Dict[str, float]:
    """
    Load existing judgment file and compute metrics.
    
    Args:
        judgment_file: Path to judgment JSONL file
        save_path: Path to save results
    
    Returns:
        Dictionary with computed metrics
    """
    
    print(f"Loading judgments from: {judgment_file}")
    
    judgments = []
    with open(judgment_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                judgments.append(json.loads(line))
    
    # Extract completions and questions
    completions = [j['judgment'] for j in judgments]
    questions = [{'question_id': j['question_id'], 'category': 'general'} for j in judgments]
    model_name = judgments[0].get('model', 'unknown') if judgments else 'unknown'
    
    # Compute metrics
    metrics = extract_ratings_and_compute_metrics(completions, questions, model_name, save_path)
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MT-bench judgments")
    parser.add_argument("--judgment-file", type=str, required=True,
                       help="Path to judgment JSONL file")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Path to save detailed results")
    
    args = parser.parse_args()
    
    if args.save_path is None:
        base_name = os.path.splitext(os.path.basename(args.judgment_file))[0]
        args.save_path = f"results/{base_name}_evaluation.json"
    
    metrics = load_and_evaluate_judgments(args.judgment_file, args.save_path)