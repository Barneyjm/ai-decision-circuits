"""
Main evaluator class for the robust evaluator.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any, List, Optional, Union, Type
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class RobustEvaluator(Runnable):
    """A configurable robust evaluator that uses multiple strategies to classify inputs."""
    
    def __init__(
        self, 
        categories: List[str],
        model: Optional[ChatAnthropic] = None,
        model_name: str = "claude-3-7-sonnet-20250219",
        parsers: Dict[str, Runnable] = None,
        combiner: Optional[Runnable] = None,
        schema_validator: Optional[Runnable] = None,
        prompt_templates: Dict[str, str] = None
    ):
        """Initialize the evaluator with configurable components.
        
        Args:
            categories: List of valid categories for classification (REQUIRED)
            model: Optional pre-configured LLM model
            model_name: Model name to use if model is not provided
            parsers: Dictionary of parser components
            combiner: Strategy combiner component
            schema_validator: Schema validator component
            prompt_templates: Dictionary of prompt templates for parsers
        """
        if not categories:
            raise ValueError("Categories must be provided and cannot be empty")
            
        # Initialize model
        if model is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
            self.model = ChatAnthropic(model=model_name)
        else:
            self.model = model
        
        # Set categories
        self.categories = categories
        
        # Set prompt templates
        self.prompt_templates = prompt_templates or {}
        
        # Set up parsers with defaults if not provided
        if parsers is None:
            from .parsers.primary_parser import PrimaryParser
            from .parsers.backup_parser import BackupParser
            from .parsers.negative_checker import NegativeChecker
            
            self.parsers = {
                "primary": PrimaryParser(
                    model=self.model, 
                    categories=self.categories,
                    prompt_template=self.prompt_templates.get("primary")
                ),
                "backup": BackupParser(
                    model=self.model, 
                    categories=self.categories,
                    prompt_template=self.prompt_templates.get("backup")
                ),
                "negative": NegativeChecker(
                    model=self.model, 
                    categories=self.categories,
                    prompt_template=self.prompt_templates.get("negative")
                )
            }
        else:
            self.parsers = parsers
        
        # Set up schema validator
        if schema_validator is None:
            from .parsers.schema_validator import SchemaValidator
            self.schema_validator = SchemaValidator(categories=self.categories)
        else:
            self.schema_validator = schema_validator
            
        # Set up combiner
        if combiner is None:
            from .combiners.strategy_combiner import StrategyCombiner
            self.combiner = StrategyCombiner()
        else:
            self.combiner = combiner
    
    def invoke(self, input_text: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process input text through all parsers and combine results.
        
        Args:
            input_text: The text to classify
            config: Optional runnable configuration
            
        Returns:
            Dictionary containing the classification result with confidence
        """
        # Run all parsers
        parser_results = {}
        for name, parser in self.parsers.items():
            parser_results[name] = parser.invoke(input_text)
        
        # Validate primary result schema
        validation_result = self.schema_validator.invoke(parser_results.get("primary", {}))
        
        # Combine results
        combined_result = self.combiner.invoke({
            "parser_results": parser_results,
            "validation_result": validation_result,
            "input_text": input_text
        })
        
        # Add debug info
        combined_result["_debug"] = {
            "parser_results": parser_results,
            "validation_result": validation_result
        }
        
        return combined_result
    
    def batch_evaluate(self, calls: List[Dict[str, Any]], limit: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate a batch of calls and calculate metrics.
        
        Args:
            calls: List of call dictionaries with 'customer_input' and 'type' fields
            limit: Optional limit on the number of calls to process
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        from .metrics.evaluation_metrics import EvaluationMetrics
        
        # Limit the number of calls if specified
        if limit is not None:
            calls = calls[:limit]
        
        # Process each call
        results = []
        for call in calls:
            customer_input = call["customer_input"]
            actual_type = call["type"]
            
            try:
                result = self.invoke(customer_input)
                
                # Create a clean copy of the result without any circular references
                clean_result = {
                    "id": call.get("id", ""),
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
                        "primary_result": debug.get("parser_results", {}).get("primary", {}).get("call_type"),
                        "backup_result": debug.get("parser_results", {}).get("backup", {}).get("call_type"),
                        "negative_check": debug.get("parser_results", {}).get("negative", ""),
                        "validation_result": debug.get("validation_result", False)
                    }
                
                results.append(clean_result)
            except Exception as e:
                print(f"Error processing call: {str(e)}")
                # Continue with the next call
        
        # Calculate metrics
        metrics_calculator = EvaluationMetrics()
        metrics = metrics_calculator.invoke({
            "calls": calls,
            "results": results,
            "categories": self.categories
        })
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RobustEvaluator":
        """Create an evaluator from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured RobustEvaluator instance
        """
        # Extract configuration values with defaults
        model_name = config.get("model_name", "claude-3-7-sonnet-20250219")
        categories = config.get("categories")
        prompt_templates = config.get("prompt_templates")
        
        if not categories:
            raise ValueError("Categories must be provided in the configuration")
        
        # Create the evaluator
        return cls(
            categories=categories,
            model_name=model_name,
            prompt_templates=prompt_templates
        )
