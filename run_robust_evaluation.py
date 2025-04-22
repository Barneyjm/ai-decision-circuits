import json
import os
import argparse
from typing import List, Dict, Any
from robust_evaluator import RobustCallClassifier, CALL_TYPES

def load_customer_calls(file_path: str) -> List[Dict[str, Any]]:
    """Load customer calls from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["calls"]

def evaluate_accuracy(calls: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluate the accuracy of the predictions."""
    total = len(calls)
    correct = sum(1 for i in range(total) if calls[i]["type"] == results[i]["call_type"])
    accuracy = correct / total if total > 0 else 0
    
    # Calculate per-category metrics
    category_metrics = {}
    for call_type in CALL_TYPES:
        true_positives = sum(1 for i in range(total) if calls[i]["type"] == call_type and results[i]["call_type"] == call_type)
        false_negatives = sum(1 for i in range(total) if calls[i]["type"] == call_type and results[i]["call_type"] != call_type)
        false_positives = sum(1 for i in range(total) if calls[i]["type"] != call_type and results[i]["call_type"] == call_type)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        category_metrics[call_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "count": true_positives + false_negatives
        }
    
    # Calculate confidence-based metrics
    confidence_metrics = {
        "high": {"count": 0, "correct": 0},
        "medium": {"count": 0, "correct": 0},
        "low": {"count": 0, "correct": 0},
        "unknown": {"count": 0, "correct": 0}
    }
    
    for i in range(total):
        confidence = results[i].get("confidence", "unknown")
        if confidence in confidence_metrics:
            confidence_metrics[confidence]["count"] += 1
            if calls[i]["type"] == results[i]["call_type"]:
                confidence_metrics[confidence]["correct"] += 1
    
    # Calculate accuracy per confidence level
    for confidence, metrics in confidence_metrics.items():
        if metrics["count"] > 0:
            metrics["accuracy"] = metrics["correct"] / metrics["count"]
        else:
            metrics["accuracy"] = 0
    
    # Calculate human review metrics
    needs_human_count = sum(1 for result in results if result.get("needs_human", False))
    needs_human_percentage = needs_human_count / total if total > 0 else 0
    
    return {
        "overall_accuracy": accuracy,
        "correct": correct,
        "total": total,
        "category_metrics": category_metrics,
        "confidence_metrics": confidence_metrics,
        "human_review": {
            "count": needs_human_count,
            "percentage": needs_human_percentage
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Run robust evaluation of LLM classification of customer calls")
    parser.add_argument("--input", default="customer-calls.json", help="Path to the customer calls JSON file")
    parser.add_argument("--output", default="robust_evaluation_results.json", help="Path to save evaluation results")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of calls to process (for testing)")
    parser.add_argument("--model", default="claude-3-7-sonnet-20250219", help="Model to use for classification")
    args = parser.parse_args()
    
    # Load customer calls
    calls = load_customer_calls(args.input)
    if args.limit:
        calls = calls[:args.limit]
    print(f"Loaded {len(calls)} customer calls")
    
    # Initialize the classifier
    classifier = RobustCallClassifier(model_name=args.model)
    print(f"Using model: {args.model}")
    
    # Classify each call
    results = []
    
    for i, call in enumerate(calls):
        print(f"Processing call {i+1}/{len(calls)}...")
        customer_input = call["customer_input"]
        actual_type = call["type"]
        
        try:
            result = classifier.classify(customer_input)
            
            # Create a clean copy of the result without any circular references
            clean_result = {
                "id": call["id"],
                "customer_input": customer_input,
                "actual_type": actual_type,
                "call_type": result["call_type"],
                "confidence": result.get("confidence", "unknown"),
                "needs_human": result.get("needs_human", False),
                "correct": actual_type == result["call_type"]
            }
            
            # Store debug info separately without any potential circular references
            if "_debug" in result:
                debug = result["_debug"]
                clean_result["debug"] = {
                    "primary_result": debug.get("primary_result", {}).get("call_type"),
                    "backup_result": debug.get("backup_result", {}).get("call_type"),
                    "negative_check": debug.get("negative_check", ""),
                    "validation_result": debug.get("validation_result", False)
                }
            
            results.append(clean_result)
            
            confidence = result.get("confidence", "unknown")
            needs_human = "Needs human review" if result.get("needs_human", False) else ""
            
            print(f"Call {i+1}: Actual={actual_type}, Predicted={result['call_type']}, " +
                  f"Confidence={confidence}, Correct={actual_type == result['call_type']} {needs_human}")
        
        except Exception as e:
            print(f"Error processing call {i+1}: {str(e)}")
            # Continue with the next call
    
    # Evaluate accuracy
    accuracy_metrics = evaluate_accuracy(calls, results)
    
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
            print(f"{call_type}: F1={metrics['f1']:.2f}, Precision={metrics['precision']:.2f}, " +
                  f"Recall={metrics['recall']:.2f}, Count={metrics['count']}")
    
    # Print confidence metrics
    print("\nConfidence-level metrics:")
    for confidence, metrics in accuracy_metrics["confidence_metrics"].items():
        if metrics["count"] > 0:
            print(f"{confidence.capitalize()}: Accuracy={metrics['accuracy']:.2%}, " +
                  f"Count={metrics['count']} ({metrics['count']/accuracy_metrics['total']:.1%})")
    
    # Print human review metrics
    human_review = accuracy_metrics["human_review"]
    print(f"\nCalls needing human review: {human_review['count']} ({human_review['percentage']:.1%})")

if __name__ == "__main__":
    main()
