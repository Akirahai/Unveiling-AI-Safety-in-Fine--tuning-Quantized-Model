# MT-bench Evaluation with Rating Extraction

This module provides automatic evaluation of MT-bench judgments with proper rating extraction and utility score computation.

## Key Features

### 1. Intelligent Rating Extraction
- **Automatic Continuation**: If a judgment doesn't contain the proper rating format `\n\nRating: [[*]]`, the system automatically continues generation by appending `\n\nRating: [[` to prompt the model to complete the rating.
- **Pattern Matching**: Extracts numeric ratings from the format `Rating: [[number]]` in the completion text.
- **Robust Parsing**: Handles multiple rating instances by taking the last/most recent one.

### 2. Comprehensive Metrics
- **Utility Score**: Average rating normalized to 0-1 scale
- **Statistical Analysis**: Mean, median, standard deviation, min/max ratings
- **Category Breakdown**: Separate metrics for different question categories (math, reasoning, coding, etc.)
- **Distribution Analysis**: Count of ratings for each score (1-10)

### 3. Flexible Usage
- **Integrated Evaluation**: Run evaluation automatically after judgment generation
- **Standalone Evaluation**: Evaluate existing judgment files
- **Batch Processing**: Handle large sets of judgments efficiently

## Usage Examples

### 1. Generate Judgments with Automatic Evaluation

```bash
python gen_judgment_vllm.py \
    --gpus 0 1 \
    --judge-model openai/gpt-oss-20b \
    --tensor-parallel-size 2 \
    --model-dir-list your-model-name \
    --evaluate \
    --evaluation-save-path results/evaluation.json
```

### 2. Evaluate Existing Judgment File

```bash
python metric.py \
    --judgment-file data/model_judgment/judgments.jsonl \
    --save-path results/evaluation.json
```

### 3. Programmatic Usage

```python
from metric import validate_judgment, load_and_evaluate_judgments

# For new judgments (during generation)
metrics = validate_judgment(
    llm=llm_model,
    sampling_params=sampling_params,
    system_prompts=system_prompts,
    user_prompts=user_prompts,
    tokenizer=tokenizer,
    questions=questions,
    model_name="your-model",
    save_path="results/evaluation.json"
)

# For existing judgment files
metrics = load_and_evaluate_judgments(
    judgment_file="judgments.jsonl",
    save_path="results/evaluation.json"
)

print(f"Utility Score: {metrics['utility_score']:.3f}")
```

## Output Format

### 1. CSV File (`*_evaluation.csv`)
```csv
question_id,category,rating,completion,model
1,general,8,"[judgment text] Rating: [[8]]",model-name
2,math,7,"[judgment text] Rating: [[7]]",model-name
```

### 2. JSON File (`*_evaluation.json`)
```json
{
  "total_questions": 80,
  "average_rating": 7.25,
  "utility_score": 0.725,
  "median_rating": 7.0,
  "std_rating": 1.8,
  "min_rating": 3.0,
  "max_rating": 10.0,
  "ratings_distribution": {
    "1": 0, "2": 1, "3": 2, ..., "10": 5
  },
  "general_avg_rating": 7.1,
  "general_utility_score": 0.71,
  "math_avg_rating": 7.5,
  "math_utility_score": 0.75
}
```

## Implementation Details

### Rating Extraction Logic
1. **Initial Generation**: Generate judgment using normal sampling parameters
2. **Format Check**: Check if completion contains `\n\nRating: [[number]]`
3. **Continuation**: If not found, append `\n\nRating: [[` and continue generation with restricted tokens
4. **Extraction**: Use regex to find and extract the numeric rating
5. **Fallback**: Assign rating of 1 if no valid rating found

### System Prompt Integration
The evaluation system properly uses system prompts from `judge_prompts.jsonl`:

```python
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}
]
formatted_prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
```

### Error Handling
- Graceful handling of missing ratings
- Validation of numeric rating ranges (1-10)
- Comprehensive logging of issues
- Fallback scoring for invalid responses

## Configuration Options

### Sampling Parameters for Evaluation
```python
eval_sampling_params = SamplingParams(
    temperature=0.0,      # Deterministic for consistent ratings
    top_p=1.0,
    max_tokens=50,        # Limited for rating completion
    stop=["]]", "\n"]     # Stop at rating end
)
```

### Customizable Paths
- Input: Judgment JSONL files
- Output: CSV (detailed) + JSON (metrics)
- Configurable save locations

This evaluation system ensures reliable extraction of MT-bench ratings and provides comprehensive metrics for model utility assessment.