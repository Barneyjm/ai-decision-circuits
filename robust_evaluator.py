import json
import os
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union
import argparse

# Load environment variables from .env file
load_dotenv()

# Define the available call types
CALL_TYPES = [
    "RESTORE", "ABATEMENT", "AMR (METERING)", "BILLING", "BPCS (BROKEN PIPE)", "BTR/O (BAD TASTE & ODOR)", 
    "C/I - DEP (CAVE IN/DEPRESSION)", "CEMENT", "CHOKED DRAIN", "CLAIMS", "COMPOST"
]

class RobustCallClassifier:
    """
    A robust classifier that uses multiple strategies to classify customer calls.
    """
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219"):
        """Initialize the classifier with the specified model."""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
        
        self.model = ChatAnthropic(model=model_name)
        self.call_types = CALL_TYPES
    
    def primary_parser(self, customer_input: str) -> Dict[str, str]:
        """
        Primary parser: Direct command with format expectations.
        """
        prompt = f"""
        Extract the category of the customer service call from the following text as a JSON object with key 'call_type'. 
        The call type must be one of: {', '.join(self.call_types)}.
        If the category cannot be determined, return {{'call_type': null}}.
        
        Customer input: "{customer_input}"
        """
        
        response = self.model.invoke(prompt)
        try:
            # Try to parse the response as JSON
            result = json.loads(response.content.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract the call type from the text
            for call_type in self.call_types:
                if call_type in response.content:
                    return {"call_type": call_type}
            return {"call_type": None}
    
    def backup_parser(self, customer_input: str) -> Dict[str, str]:
        """
        Backup parser: Chain of thought approach with formatting instructions.
        """
        prompt = f"""
        First, identify the main issue or concern in the customer's message.
        Then, match it to one of the following categories: {', '.join(self.call_types)}.
        
        Think through each category and determine which one best fits the customer's issue.
        
        Return your answer as a JSON object with key 'call_type'.
        
        Customer input: "{customer_input}"
        """
        
        response = self.model.invoke(prompt)
        try:
            # Try to parse the response as JSON
            result = json.loads(response.content.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract the call type from the text
            for call_type in self.call_types:
                if call_type in response.content:
                    return {"call_type": call_type}
            return {"call_type": None}
    
    def negative_checker(self, customer_input: str) -> str:
        """
        Negative checker: Determines if the text contains enough information to categorize.
        """
        prompt = f"""
        Does this customer service call contain enough information to categorize it into one of these types: 
        {', '.join(self.call_types)}?
        
        Answer only 'yes' or 'no'.
        
        Customer input: "{customer_input}"
        """
        
        response = self.model.invoke(prompt)
        answer = response.content.strip().lower()
        
        if "yes" in answer:
            return "yes"
        elif "no" in answer:
            return "no"
        else:
            # Default to yes if the answer is unclear
            return "yes"
    
    @staticmethod
    def validate_call_type(parsed_output: Dict[str, Any]) -> bool:
        """
        Schema validator: Checks if the output matches the expected schema.
        """
        # Check if output matches expected schema
        if not isinstance(parsed_output, dict) or 'call_type' not in parsed_output:
            return False
            
        # Verify the extracted call type is in our list of known types or null
        call_type = parsed_output['call_type']
        return call_type is None or call_type in CALL_TYPES
    
    @staticmethod
    def combine_results(
        primary_result: Dict[str, str], 
        backup_result: Dict[str, str], 
        negative_check: str, 
        validation_result: bool,
        customer_input: str
    ) -> Dict[str, str]:
        """
        Combiner: Combines the results from different strategies.
        """
        # If validation failed, use backup
        if not validation_result:
            if RobustCallClassifier.validate_call_type(backup_result):
                return backup_result
            else:
                return {"call_type": None, "confidence": "low", "needs_human": True}
                
        # If negative check says no call type can be determined but we extracted one, double-check
        if negative_check == 'no' and primary_result['call_type'] is not None:
            if backup_result['call_type'] is None:
                return {'call_type': None, "confidence": "low", "needs_human": True}
            elif backup_result['call_type'] == primary_result['call_type']:
                # Both agree despite negative check, so go with it but mark low confidence
                return {'call_type': primary_result['call_type'], "confidence": "medium"}
            else:
                return {"call_type": None, "confidence": "low", "needs_human": True}
                
        # If primary and backup agree, high confidence
        if primary_result['call_type'] == backup_result['call_type'] and primary_result['call_type'] is not None:
            return {'call_type': primary_result['call_type'], "confidence": "high"}
            
        # Default: use primary result with medium confidence
        if primary_result['call_type'] is not None:
            return {'call_type': primary_result['call_type'], "confidence": "medium"}
        else:
            return {'call_type': None, "confidence": "low", "needs_human": True}
    
    def classify(self, customer_input: str) -> Dict[str, Any]:
        """
        Classify the customer input using the robust approach.
        """
        # Run all classifiers
        primary_result = self.primary_parser(customer_input)
        backup_result = self.backup_parser(customer_input)
        negative_check = self.negative_checker(customer_input)
        validation_result = self.validate_call_type(primary_result)
        
        # Combine results
        result = self.combine_results(
            primary_result, 
            backup_result, 
            negative_check, 
            validation_result,
            customer_input
        )
        
        # Add the original results for debugging
        result["_debug"] = {
            "primary_result": primary_result,
            "backup_result": backup_result,
            "negative_check": negative_check,
            "validation_result": validation_result
        }
        
        return result


def load_customer_calls(file_path: str) -> List[Dict[str, Any]]:
    """Load customer calls from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data["calls"]


def main():
    """
    Example usage of the RobustCallClassifier.
    """
    parser = argparse.ArgumentParser(description="Test the RobustCallClassifier on a few examples")
    parser.add_argument("--input", default="customer-calls.json", help="Path to the customer calls JSON file")
    parser.add_argument("--limit", type=int, default=5, help="Number of calls to process")
    args = parser.parse_args()
    
    # Create a classifier
    classifier = RobustCallClassifier()
    
    # Load customer calls from the JSON file
    calls = load_customer_calls(args.input)
    if args.limit:
        calls = calls[:args.limit]
    
    print(f"Testing classifier on {len(calls)} examples from {args.input}")
    
    # Classify each call
    for i, call in enumerate(calls):
        customer_input = call["customer_input"]
        actual_type = call["type"]
        
        print(f"\nExample {i+1}:")
        print(f"Customer input: {customer_input}")
        print(f"Actual type: {actual_type}")
        
        result = classifier.classify(customer_input)
        
        print(f"Classification: {result['call_type']}")
        print(f"Confidence: {result.get('confidence', 'unknown')}")
        print(f"Correct: {actual_type == result['call_type']}")
        
        if result.get('needs_human', False):
            print("This call needs human review.")
        
        # Print debug info
        debug = result.get("_debug", {})
        print("\nDebug info:")
        print(f"Primary result: {debug.get('primary_result', {})}")
        print(f"Backup result: {debug.get('backup_result', {})}")
        print(f"Negative check: {debug.get('negative_check', '')}")
        print(f"Validation result: {debug.get('validation_result', '')}")


if __name__ == "__main__":
    main()
