"""
Negative checker that determines if the text contains enough information to categorize.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from langchain_anthropic import ChatAnthropic
from typing import Dict, Any, List, Optional

class NegativeChecker(Runnable):
    """Negative checker that determines if the text contains enough information to categorize."""
    
    def __init__(
        self, 
        model: ChatAnthropic,
        categories: List[str],
        prompt_template: Optional[str] = None
    ):
        """Initialize the negative checker.
        
        Args:
            model: The LLM model to use for checking
            categories: List of valid categories for classification
            prompt_template: Optional custom prompt template
        """
        self.model = model
        self.categories = categories
        self.prompt_template = prompt_template or """
        Does this customer service call contain enough information to categorize it into one of these types: 
        {categories}?
        
        Answer only 'yes' or 'no'.
        
        Customer input: "{input_text}"
        """
    
    def invoke(self, input_text: str, config: Optional[RunnableConfig] = None) -> str:
        """Process input through the negative checker.
        
        Args:
            input_text: The text to check
            config: Optional runnable configuration
            
        Returns:
            "yes" or "no" indicating if the text can be categorized
        """
        prompt = self.prompt_template.format(
            categories=", ".join(self.categories),
            input_text=input_text
        )
        
        response = self.model.invoke(prompt)
        answer = response.content.strip().lower()
        
        if "yes" in answer:
            return "yes"
        elif "no" in answer:
            return "no"
        else:
            # Default to yes if the answer is unclear
            return "yes"
