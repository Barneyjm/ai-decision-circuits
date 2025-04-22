import json
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Define the available call types
CALL_TYPES = [
    "RESTORE", "ABATEMENT", "AMR (METERING)", "BILLING", "BPCS (BROKEN PIPE)", "BTR/O (BAD TASTE & ODOR)", 
    "C/I - DEP (CAVE IN/DEPRESSION)", "CEMENT", "CHOKED DRAIN", "CLAIMS", "COMPOST"
]


def load_customer_calls(file_path: str) -> List[Dict[str, Any]]:
    """Load customer calls from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["calls"]

def classify_with_anthropic(customer_input: str) -> str:
    """
    Classify customer input using Anthropic's Claude API.
    
    Returns the predicted call type from the available types.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    
    # Initialize the ChatAnthropic model
    # model = ChatAnthropic(model='claude-3-7-sonnet-20250219')
    model = ChatAnthropic(model='claude-3-5-sonnet-latest')
    
    # Create a prompt that asks the model to classify the customer input
    prompt = f"""
    You are a customer service AI for a water utility company. Classify the following customer input into one of these categories:
    {', '.join(CALL_TYPES)}
    
    Customer input: "{customer_input}"
    
    Respond with just the category name, nothing else.
    """
    
    # Get the response from Claude
    response = model.invoke(prompt)
    predicted_type = response.content.strip()
    
    # Ensure the predicted type is one of the valid types
    for call_type in CALL_TYPES:
        if call_type in predicted_type:
            return call_type
    
    # If no exact match, return the closest match
    return predicted_type

def evaluate_accuracy(calls: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
    """Evaluate the accuracy of the predictions."""
    total = len(calls)
    correct = sum(1 for i in range(total) if calls[i]["type"] == predictions[i])
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-category metrics
    category_metrics = {}
    for call_type in CALL_TYPES:
        true_positives = sum(1 for i in range(total) if calls[i]["type"] == call_type and predictions[i] == call_type)
        false_negatives = sum(1 for i in range(total) if calls[i]["type"] == call_type and predictions[i] != call_type)
        false_positives = sum(1 for i in range(total) if calls[i]["type"] != call_type and predictions[i] == call_type)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[call_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": true_positives + false_negatives
        }
    
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_metrics": category_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM classification of customer calls")
    parser.add_argument("--input", default="customer-calls.json", help="Path to the customer calls JSON file")
    parser.add_argument("--output", default="evaluation_results.json", help="Path to save evaluation results")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of calls to process (for testing)")
    args = parser.parse_args()
    
    # Load customer calls
    calls = load_customer_calls(args.input)
    if args.limit:
        calls = calls[:args.limit]
    print(f"Loaded {len(calls)} customer calls")
    
    # Classify each call
    predictions = []
    results = []
    
    for i, call in enumerate(calls):
        print(f"Processing call {i+1}/{len(calls)}...")
        customer_input = call["customer_input"]
        actual_type = call["type"]
        
        try:
            predicted_type = classify_with_anthropic(customer_input)
            predictions.append(predicted_type)
            
            result = {
                "id": call["id"],
                "customer_input": customer_input,
                "actual_type": actual_type,
                "predicted_type": predicted_type,
                "correct": actual_type == predicted_type
            }
            results.append(result)
            
            print(f"Call {i+1}: Actual={actual_type}, Predicted={predicted_type}, Correct={actual_type == predicted_type}")
        
        except Exception as e:
            print(f"Error processing call {i+1}: {str(e)}")
            # Continue with the next call
    
    # Evaluate accuracy
    accuracy_metrics = evaluate_accuracy(calls, predictions)
    
    # Save results
    output_data = {
        "results": results,
        "metrics": accuracy_metrics
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nEvaluation complete. Results saved to {args.output}")
    print(f"Overall accuracy: {accuracy_metrics['overall_accuracy']:.2%} ({accuracy_metrics['correct']}/{accuracy_metrics['total']})")
    
    # Print per-category metrics
    print("\nPer-category metrics:")
    for call_type, metrics in accuracy_metrics["category_metrics"].items():
        if metrics["count"] > 0:
            print(f"{call_type}: F1={metrics['f1']:.2f}, Precision={metrics['precision']:.2f}, Recall={metrics['recall']:.2f}, Count={metrics['count']}")

if __name__ == "__main__":
    main()
