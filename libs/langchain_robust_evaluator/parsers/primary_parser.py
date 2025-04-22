"""
Primary parser that uses direct command with format expectations.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any, List, Optional
import json

class PrimaryParser(Runnable):
    """Primary parser that uses direct command with format expectations."""
    
    def __init__(
        self, 
        model: ChatAnthropic,
        categories: List[str],
        prompt_template: Optional[str] = None
    ):
        """Initialize the primary parser.
        
        Args:
            model: The LLM model to use for parsing
            categories: List of valid categories for classification
            prompt_template: Optional custom prompt template
        """
        self.model = model
        self.categories = categories
        self.prompt_template = prompt_template or """
        Extract the category of the customer service call from the following text as a JSON object with key 'call_type'. 
        The call type must be one of: {categories}.
        If the category cannot be determined, return {{'call_type': null}}.
        
        Customer input: "{input_text}"
        """
    
    def invoke(self, input_text: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process input through the primary parser.
        
        Args:
            input_text: The text to classify
            config: Optional runnable configuration
            
        Returns:
            Dictionary containing the classification result
        """
        prompt = self.prompt_template.format(
            categories=", ".join(self.categories),
            input_text=input_text
        )
        
        response = self.model.invoke(prompt)
        try:
            # Try to parse the response as JSON
            result = json.loads(response.content.strip())
            return result
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract the call type from the text
            for call_type in self.categories:
                if call_type in response.content:
                    return {"call_type": call_type}
            return {"call_type": None}
