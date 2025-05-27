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

    def amplify(self, input_text: str, amplification_factor: float = 1.5, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process input text through parsers with an amplification step and combine results.
        
        Args:
            input_text: The text to classify
            amplification_factor: Factor to determine the intensity of amplification
            config: Optional runnable configuration
            
        Returns:
            Dictionary containing the classification result with confidence and amplification info
        """
        # Run all parsers to get base results
        parser_results = {}
        for name, parser in self.parsers.items():
            parser_results[name] = parser.invoke(input_text)

        # Determine amplification instruction based on the factor
        if amplification_factor > 2.0:
            amplification_instruction = "Amplify your response significantly, providing extensive details and expressing strong confidence."
        elif amplification_factor > 1.0:
            amplification_instruction = f"Amplify your response, providing more detail and expressing higher confidence by a factor of {amplification_factor}."
        elif amplification_factor == 1.0:
            amplification_instruction = "Provide your standard response." # No real amplification
        else: # amplification_factor < 1.0, potentially de-amplification or slight amplification
            amplification_instruction = f"Slightly amplify your response, providing a bit more detail. Factor: {amplification_factor}."

        # Construct the modified prompt for the primary parser
        # Assuming the primary parser can handle a prompt dictionary or has a default way to take instructions
        amplified_primary_input = {
            "input": input_text, 
            "instruction": amplification_instruction,
            # If the primary parser expects a specific prompt structure, this needs to be adapted.
            # For now, sending as a dictionary, assuming the parser can handle it.
            # Alternatively, we might need to access the parser's prompt_template and modify it.
            # For simplicity, let's assume the primary parser can take a modified input like this.
            # If not, we would need to get the original prompt and append the instruction.
            "original_prompt_text_key_if_needed": f"{input_text}\n\nInstruction: {amplification_instruction}"
        }
        
        # Re-invoke the primary parser with the modified prompt.
        # This is a simplification. The actual re-invocation might be more complex
        # depending on how PrimaryParser is implemented and if it supports direct instruction injection.
        # If PrimaryParser uses a fixed prompt template, we might need a more sophisticated way
        # to modify the prompt, or the PrimaryParser itself would need an 'amplify' mode.
        
        # For now, let's assume the primary parser is the first one in self.parsers dictionary (e.g., self.parsers["primary"])
        # And it can be invoked with the modified input.
        if "primary" in self.parsers:
            # This assumes the primary parser's invoke method can handle a dictionary
            # or a string that now includes the instruction.
            # If the primary parser expects just a string, we'd use:
            # amplified_prompt_text = f"{input_text}\n\nInstruction: {amplification_instruction}"
            # parser_results["primary_amplified"] = self.parsers["primary"].invoke(amplified_prompt_text)
            # For now, let's stick to a more structured input if possible, assuming PrimaryParser is flexible.
            # If it strictly expects the same input format as other parsers (just input_text),
            # then this approach will need to be revised.
            # A common way is to have a specific key in the input dict that the parser looks for.
            
            # Let's assume the PrimaryParser can be re-invoked with the original input_text,
            # and the amplification instruction is handled internally by the parser if it's designed for it,
            # or by passing a modified prompt.
            # For this exercise, we'll assume the primary parser takes the original input text,
            # and we need to create a modified prompt string.
            
            # Get the original prompt for the primary parser, if available
            primary_parser_prompt_template = self.prompt_templates.get(
                "primary", 
                "Classify the following text: {input_text}\nCategories: {categories}" # A default fallback
            )
            
            # Construct the amplified prompt text
            # Ensure categories are passed if the template requires them
            formatted_categories = ", ".join(self.categories)
            amplified_prompt_text = primary_parser_prompt_template.format(
                input_text=input_text, 
                categories=formatted_categories
            ) + f"\n\n{amplification_instruction}"

            # Re-invoke the primary parser with the amplified prompt
            amplified_primary_result = self.parsers["primary"].invoke(amplified_prompt_text)
            parser_results["primary_amplified"] = amplified_primary_result
        else:
            # Fallback if primary parser is not found, though it should be there
            parser_results["primary_amplified"] = parser_results.get("primary", {})


        # Validate the amplified primary result schema (or the original primary if amplification failed)
        # The schema validator should ideally run on the result that will be used by the combiner.
        # If 'primary_amplified' is the new primary, it should be validated.
        validation_result = self.schema_validator.invoke(parser_results.get("primary_amplified", parser_results.get("primary", {})))
        
        # Combine results - the combiner needs to be aware of the 'primary_amplified' result
        # Or we can replace the 'primary' result with the 'primary_amplified' one before combining.
        # Let's pass all results and let the combiner decide, or update the primary result.
        
        # Option 1: Let combiner handle it (requires combiner modification)
        # combined_input_to_combiner = {
        #     "parser_results": parser_results, # now includes "primary_amplified"
        #     "validation_result": validation_result,
        #     "input_text": input_text,
        #     "amplification_factor": amplification_factor
        # }

        # Option 2: Replace primary result before combining (simpler for existing combiner)
        # Create a mutable copy of parser_results for the combiner if necessary
        results_for_combiner = parser_results.copy()
        if "primary_amplified" in results_for_combiner:
            results_for_combiner["primary"] = results_for_combiner["primary_amplified"] # Overwrite primary with amplified

        combined_result = self.combiner.invoke({
            "parser_results": results_for_combiner, # Use the (potentially) updated results
            "validation_result": validation_result,
            "input_text": input_text
        })
        
        # Add debug and amplification info
        combined_result["_debug"] = {
            "parser_results": parser_results, # contains original and amplified primary
            "validation_result": validation_result,
            "amplification_factor": amplification_factor,
            "amplification_instruction": amplification_instruction
        }
        if "primary_amplified" in parser_results: # Add more specific debug for amplified output
            combined_result["_debug"]["amplified_primary_output"] = parser_results["primary_amplified"]

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
