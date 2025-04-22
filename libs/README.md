# LangChain Robust Evaluator

A configurable LangChain package for robust LLM evaluation using multiple strategies to improve classification reliability.

## Overview

The LangChain Robust Evaluator provides a sophisticated approach to evaluating LLM performance on classification tasks. Unlike simple single-pass evaluations, this package employs multiple strategies to improve classification reliability:

1. **Primary Parser**: Direct command with format expectations
2. **Backup Parser**: Chain of thought approach for more complex reasoning
3. **Negative Checker**: Determines if text contains enough information to categorize
4. **Schema Validator**: Checks if output matches expected schema

Results from these strategies are combined to produce:
- Final classification with confidence levels (high, medium, low)
- Flags for calls that need human review
- Comprehensive evaluation metrics

## Installation

```bash
pip install langchain-robust-evaluator
```

Or install from source:

```bash
git clone https://github.com/yourusername/langchain-robust-evaluator.git
cd langchain-robust-evaluator
pip install -e .
```

## Requirements

- Python 3.8+
- langchain>=0.1.0
- langchain-anthropic>=0.3.10
- python-dotenv>=1.0.0

## Quick Start

```python
from langchain_robust_evaluator import RobustEvaluator

# Define your classification categories
categories = ["BILLING", "TECHNICAL", "SALES", "ACCOUNT", "COMPLAINT"]

# Initialize with required categories
evaluator = RobustEvaluator(categories=categories)

# Classify a single input
result = evaluator.invoke("I need help with my monthly bill payment")
print(f"Classification: {result['call_type']}, Confidence: {result['confidence']}")

# Check if human review is needed
if result.get("needs_human", False):
    print("This message needs human review")
```

## Features

### Configurable Components

Every aspect of the evaluation system can be customized:

- **LLM Model**: Use any LangChain-compatible LLM
- **Categories**: Define your own classification categories (required)
- **Prompt Templates**: Customize prompts for each parser
- **Parsers**: Add or replace parsing strategies
- **Combiners**: Implement custom logic for combining results

### Confidence Levels

Results include confidence levels to help prioritize review:

- **High**: Strong agreement between strategies
- **Medium**: Reasonable confidence but some disagreement
- **Low**: Significant uncertainty, may need human review

### Comprehensive Metrics

When evaluating batches of inputs, the system generates detailed metrics:

- Overall accuracy
- Per-category performance (precision, recall, F1 score)
- Confidence-level metrics (accuracy by confidence level)
- Statistics on calls flagged for human review

## Advanced Usage

### Custom Categories

Categories are required and must be specified when creating the evaluator:

```python
# Define categories for customer service
categories = ["BILLING", "TECHNICAL", "ACCOUNT", "SALES", "COMPLAINT"]

# Create evaluator with categories
evaluator = RobustEvaluator(categories=categories)

# Define categories for product feedback
product_categories = ["QUESTION", "FEEDBACK", "BUG_REPORT", "FEATURE_REQUEST"]

# Create evaluator for product feedback
product_evaluator = RobustEvaluator(categories=product_categories)
```

### Custom Prompt Templates

```python
# Define custom prompts
prompts = {
    "primary": "Classify this message into one of these categories: {categories}. Message: \"{input_text}\"",
    "backup": "Think step by step about what the user is asking for. Then classify it into one of these categories: {categories}."
}

# Create evaluator with custom prompts
evaluator = RobustEvaluator(
    categories=["INQUIRY", "SUPPORT", "FEEDBACK", "SALES"],
    prompt_templates=prompts
)
```

### Batch Evaluation

```python
# Load your data
messages = [
    {"customer_input": "I need help with my monthly bill", "type": "BILLING"},
    {"customer_input": "My service is not working properly", "type": "TECHNICAL"}
]

# Run batch evaluation
results = evaluator.batch_evaluate(messages)

# Access metrics
metrics = results["metrics"]
print(f"Overall accuracy: {metrics['overall_accuracy']:.2%}")
```

### Configuration-based Creation

```python
config = {
    "model_name": "claude-3-7-sonnet-20250219",
    "categories": ["INQUIRY", "SUPPORT", "FEEDBACK", "SALES"],
    "prompt_templates": {
        "primary": "Classify this message into one of these categories: {categories}. Message: \"{input_text}\""
    }
}

evaluator = RobustEvaluator.from_config(config)
```

## Integration with LangChain

The package integrates seamlessly with other LangChain components:

```python
from langchain.schema.runnable import RunnablePassthrough
from langchain_robust_evaluator import RobustEvaluator

# Define categories
categories = ["INQUIRY", "SUPPORT", "FEEDBACK", "SALES"]

# Create a pipeline
pipeline = (
    {"input": RunnablePassthrough()} 
    | {"classification": RobustEvaluator(categories=categories), "input": RunnablePassthrough()}
    | (lambda x: {"input": x["input"], "result": x["classification"]})
)

# Process multiple inputs
results = pipeline.batch(["I need help with billing", "I'm interested in your enterprise plan"])
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
