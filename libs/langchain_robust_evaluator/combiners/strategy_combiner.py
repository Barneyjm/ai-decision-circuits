"""
Combines results from different parsing strategies.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from typing import Dict, Any, Optional

class StrategyCombiner(Runnable):
    """Combines results from different parsing strategies."""
    
    def __init__(self, confidence_thresholds: Optional[Dict[str, float]] = None):
        """Initialize the strategy combiner.
        
        Args:
            confidence_thresholds: Optional dictionary of confidence thresholds
        """
        self.confidence_thresholds = confidence_thresholds or {
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3
        }
    
    def invoke(self, inputs: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Combine parser results into a final classification with confidence.
        
        Args:
            inputs: Dictionary containing parser results, validation result, and input text
            config: Optional runnable configuration
            
        Returns:
            Dictionary containing the final classification with confidence level
        """
        parser_results = inputs["parser_results"]
        validation_result = inputs["validation_result"]
        
        primary_result = parser_results.get("primary", {"call_type": None})
        backup_result = parser_results.get("backup", {"call_type": None})
        negative_check = parser_results.get("negative", "yes")
        
        # If validation failed, use backup
        if not validation_result:
            if backup_result.get("call_type") is not None:
                return {
                    "call_type": backup_result["call_type"],
                    "confidence": "medium"
                }
            else:
                return {
                    "call_type": None, 
                    "confidence": "low", 
                    "needs_human": True
                }
                
        # If negative check says no call type can be determined but we extracted one, double-check
        if negative_check == 'no' and primary_result.get('call_type') is not None:
            if backup_result.get('call_type') is None:
                return {
                    'call_type': None, 
                    "confidence": "low", 
                    "needs_human": True
                }
            elif backup_result.get('call_type') == primary_result.get('call_type'):
                # Both agree despite negative check, so go with it but mark medium confidence
                return {
                    'call_type': primary_result.get('call_type'), 
                    "confidence": "medium"
                }
            else:
                return {
                    "call_type": None, 
                    "confidence": "low", 
                    "needs_human": True
                }
                
        # If primary and backup agree, high confidence
        if primary_result.get('call_type') == backup_result.get('call_type') and primary_result.get('call_type') is not None:
            return {
                'call_type': primary_result.get('call_type'), 
                "confidence": "high"
            }
            
        # Default: use primary result with medium confidence
        if primary_result.get('call_type') is not None:
            return {
                'call_type': primary_result.get('call_type'), 
                "confidence": "medium"
            }
        else:
            return {
                'call_type': None, 
                "confidence": "low", 
                "needs_human": True
            }
