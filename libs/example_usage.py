"""
Example usage of the LangChain Robust Evaluator package.
"""

import json
import os
from langchain_robust_evaluator import RobustEvaluator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_sample_data(file_path: str = None):
    """Load sample data or create it if file doesn't exist."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data["calls"]
    else:
        # Return generic sample data
        return [
            {
                "id": "1",
                "customer_input": "I need help with my monthly bill payment",
                "type": "BILLING"
            },
            {
                "id": "2",
                "customer_input": "My service is not working properly",
                "type": "TECHNICAL"
            },
            {
                "id": "3",
                "customer_input": "I want to upgrade my subscription",
                "type": "SALES"
            },
            {
                "id": "4",
                "customer_input": "How do I reset my password?",
                "type": "ACCOUNT"
            },
            {
                "id": "5",
                "customer_input": "I have a complaint about the service quality",
                "type": "COMPLAINT"
            }
        ]

def extract_categories_from_data(calls):
    """Extract unique categories from the data."""
    return sorted(list(set(call["type"] for call in calls)))

def main():
    # Define generic categories for customer service classification
    generic_categories = [
        "BILLING", "TECHNICAL", "SALES", "ACCOUNT", "COMPLAINT"
    ]
    
    # Load sample data
    try:
        calls = load_sample_data("../customer-calls.json")
        # Extract categories from the actual data
        water_utility_categories = extract_categories_from_data(calls)
    except:
        calls = load_sample_data()
        water_utility_categories = generic_categories
    
    print(f"Loaded {len(calls)} sample calls")
    
    # Example 1: Basic usage with required categories
    print("\n--- Example 1: Basic Usage ---")
    # Use the appropriate categories based on the data source
    if "RESTORE" in water_utility_categories:  # Check if we're using water utility data
        evaluator = RobustEvaluator(categories=water_utility_categories)
    else:
        evaluator = RobustEvaluator(categories=generic_categories)
    
    # Classify a single call
    sample_call = calls[0]
    result = evaluator.invoke(sample_call["customer_input"])
    print(f"Input: {sample_call['customer_input']}")
    print(f"Actual type: {sample_call['type']}")
    print(f"Predicted type: {result['call_type']}")
    print(f"Confidence: {result.get('confidence', 'unknown')}")
    print(f"Needs human review: {result.get('needs_human', False)}")
    
    # Example 2: Custom configuration
    print("\n--- Example 2: Custom Configuration ---")
    # Define custom categories for a different domain
    custom_categories = [
        "QUESTION", "FEEDBACK", "BUG_REPORT", "FEATURE_REQUEST"
    ]
    
    # Define custom prompt templates
    custom_prompts = {
        "primary": """
        Classify the following user message into one of these categories: {categories}.
        Return a JSON object with the format: {{"call_type": "CATEGORY_NAME"}}
        
        User message: "{input_text}"
        """,
        "backup": """
        Think step by step about what the user is asking for.
        Then classify it into one of these categories: {categories}.
        Return a JSON with the key 'call_type' and the category as the value.
        
        User message: "{input_text}"
        """
    }
    
    # Create custom evaluator
    custom_evaluator = RobustEvaluator(
        categories=custom_categories,
        model_name="claude-3-7-sonnet-20250219",
        prompt_templates=custom_prompts
    )
    
    # Sample product feedback messages
    product_feedback = [
        "How do I integrate this with my existing system?",
        "The new UI is much better than the previous version!",
        "The app crashes whenever I try to upload a file",
        "It would be great if you could add dark mode"
    ]
    
    # Classify a sample message
    sample_message = product_feedback[0]
    custom_result = custom_evaluator.invoke(sample_message)
    print(f"Input: {sample_message}")
    print(f"Custom categories: {custom_categories}")
    print(f"Predicted type: {custom_result['call_type']}")
    print(f"Confidence: {custom_result.get('confidence', 'unknown')}")
    
    # Example 3: Batch evaluation with metrics
    print("\n--- Example 3: Batch Evaluation ---")
    print(f"Using categories: {water_utility_categories[:5]}...")
    
    # Create an evaluator with the correct categories for batch evaluation
    batch_evaluator = RobustEvaluator(categories=water_utility_categories)
    
    # Prepare sample data in the right format
    evaluation_data = []
    for i, call in enumerate(calls[:5]):
        evaluation_data.append({
            "id": str(i+1),
            "customer_input": call["customer_input"],
            "type": call["type"]
        })
    
    # Run batch evaluation
    evaluation_results = batch_evaluator.batch_evaluate(evaluation_data)
    
    # Print summary metrics
    metrics = evaluation_results["metrics"]
    print(f"Overall accuracy: {metrics['overall_accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
    
    # Print confidence metrics
    print("\nConfidence-level metrics:")
    for confidence, conf_metrics in metrics["confidence_metrics"].items():
        if conf_metrics["count"] > 0:
            print(f"{confidence.capitalize()}: Accuracy={conf_metrics['accuracy']:.2%}, " +
                  f"Count={conf_metrics['count']} ({conf_metrics['count']/metrics['total']:.1%})")
    
    # Print human review metrics
    human_review = metrics["human_review"]
    print(f"\nCalls needing human review: {human_review['count']} ({human_review['percentage']:.1%})")
    
    # Example 4: Creating from configuration
    print("\n--- Example 4: Creating from Configuration ---")
    config = {
        "model_name": "claude-3-7-sonnet-20250219",
        "categories": ["INQUIRY", "SUPPORT", "FEEDBACK", "SALES"],
        "prompt_templates": {
            "primary": "Classify this message into one of these categories: {categories}. Message: \"{input_text}\""
        }
    }
    
    config_evaluator = RobustEvaluator.from_config(config)
    print(f"Created evaluator with {len(config['categories'])} categories: {config['categories']}")
    
    # Classify a message with the config-based evaluator
    sample_message = "I'm interested in your enterprise pricing options"
    config_result = config_evaluator.invoke(sample_message)
    print(f"Input: {sample_message}")
    print(f"Predicted type: {config_result['call_type']}")
    print(f"Confidence: {config_result.get('confidence', 'unknown')}")

if __name__ == "__main__":
    main()
