# Customer Call Categorization Evaluator

This script evaluates an LLM's ability to categorize customer service calls for a water utility company based on the customer's input text.

## Overview

The script:
1. Reads customer call data from a JSON file
2. Sends each customer input to Anthropic's Claude model for classification
3. Compares the LLM's prediction with the actual category
4. Generates detailed evaluation metrics including accuracy, precision, recall, and F1 score

## Available Categories

The script evaluates classification into these categories:
- RESTORE
- ABATEMENT
- AMR
- BILLING
- BPCS
- BTR/O
- C/I - DEP
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
   cp .env.example .env
   ```
   Then edit the `.env` file to add your actual Anthropic API key.

## Usage

Run the script with:
```
python evaluate_calls.py
```

Optional arguments:
- `--input`: Path to the customer calls JSON file (default: "customer-calls.json")
- `--output`: Path to save evaluation results (default: "evaluation_results.json")

Example:
```
python evaluate_calls.py --input my_calls.json --output my_results.json
```

## Output

The script generates a JSON file with:
- Individual results for each call
- Overall accuracy metrics
- Per-category metrics (precision, recall, F1 score)

It also prints a summary of the results to the console.
