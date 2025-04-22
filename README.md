# Customer Call Categorization Evaluator

This project evaluates an LLM's ability to categorize customer service calls for a water utility company based on the customer's input text, using a robust multi-strategy approach.

## Overview

The project includes two main evaluation approaches:

### Basic Evaluation
- Reads customer call data from a JSON file
- Sends each customer input to Anthropic's Claude model for classification
- Compares the LLM's prediction with the actual category
- Generates basic evaluation metrics

### Robust Evaluation
The robust evaluation system uses multiple strategies to improve classification reliability:

1. **Primary Parser**: Direct command with format expectations
2. **Backup Parser**: Chain of thought approach for more complex reasoning
3. **Negative Checker**: Determines if text contains enough information to categorize
4. **Schema Validator**: Checks if output matches expected schema

Results from these strategies are combined to produce:
- Final classification with confidence levels (high, medium, low)
- Flags for calls that need human review
- Comprehensive evaluation metrics

## Available Categories

The system evaluates classification into these categories:
- RESTORE
- ABATEMENT
- AMR (METERING)
- BILLING
- BPCS (BROKEN PIPE)
- BTR/O (BAD TASTE & ODOR)
- C/I - DEP (CAVE IN/DEPRESSION)
- CEMENT
- CHOKED DRAIN
- CLAIMS
- COMPOST

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Anthropic API key:
   ```
   ANTHROPIC_API_KEY=your_api_key_here
   ```

## Usage

### Running Basic Evaluation
```
python evaluate.py --input customer-calls.json --output evaluation_results.json
```

### Running Robust Evaluation
```
python run_robust_evaluation.py --input customer-calls.json --output robust_evaluation_results.json --model claude-3-7-sonnet-20250219
```

### Command-line Arguments
- `--input`: Path to the customer calls JSON file (default: "customer-calls.json")
- `--output`: Path to save evaluation results (default: "evaluation_results.json" or "robust_evaluation_results.json")
- `--limit`: Limit the number of calls to process (optional, for testing)
- `--model`: Model to use for classification (default: "claude-3-7-sonnet-20250219")

## Evaluation Metrics

The robust evaluation generates comprehensive metrics including:
- Overall accuracy
- Per-category performance (precision, recall, F1 score)
- Confidence-level metrics (accuracy by confidence level)
- Statistics on calls flagged for human review
