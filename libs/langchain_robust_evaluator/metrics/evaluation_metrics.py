"""
Calculates evaluation metrics for classification results.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from typing import Dict, Any, List, Optional

class EvaluationMetrics(Runnable):
    """Calculates evaluation metrics for classification results."""
    
    def invoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Calculate metrics from classification results.
        
        Args:
            inputs: Dictionary containing calls and results
            config: Optional runnable configuration
            
        Returns:
            Dictionary containing evaluation metrics
        """
        calls = inputs["calls"]
        results = inputs["results"]
        categories = inputs.get("categories", [])
        
        total = len(calls)
        correct = sum(1 for i in range(total) if calls[i]["type"] == results[i]["call_type"])
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-category metrics
        category_metrics = {}
        for call_type in categories:
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
