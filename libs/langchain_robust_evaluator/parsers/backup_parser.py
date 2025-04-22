"""
Backup parser that uses a chain of thought approach.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any, List, Optional
import json

class BackupParser(Runnable):
    """Backup parser that uses chain of thought approach with formatting instructions."""
    
    def __init__(
        self, 
        model: ChatAnthropic,
        categories: List[str],
        prompt_template: Optional[str] = None
    ):
        """Initialize the backup parser.
        
        Args:
            model: The LLM model to use for parsing
            categories: List of valid categories for classification
            prompt_template: Optional custom prompt template
        """
        self.model = model
        self.categories = categories
        self.prompt_template = prompt_template or """
        First, identify the main issue or concern in the customer's message.
        Then, match it to one of the following categories: {categories}.
        
        Think through each category and determine which one best fits the customer's issue.
        
        Return your answer as a JSON object with key 'call_type'.
        
        Customer input: "{input_text}"
        """
    
    def invoke(self, input_text: str, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        """Process input through the backup parser.
        
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
