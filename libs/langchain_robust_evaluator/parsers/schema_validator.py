"""
Schema validator that checks if the output matches the expected schema.
"""

from langchain.schema.runnable import Runnable, RunnableConfig
from typing import Dict, Any, List, Optional

class SchemaValidator(Runnable):
    """Schema validator that checks if the output matches the expected schema."""
    
    def __init__(self, categories: List[str]):
        """Initialize the schema validator.
        
        Args:
            categories: List of valid categories for classification
        """
        self.categories = categories
    
    def invoke(self, parsed_output: Dict[str, Any], config: Optional[RunnableConfig] = None) -> bool:
        """Validate if the parsed output matches the expected schema.
        
        Args:
            parsed_output: The output to validate
            config: Optional runnable configuration
            
        Returns:
            Boolean indicating if the output is valid
        """
        # Check if output matches expected schema
        if not isinstance(parsed_output, dict) or 'call_type' not in parsed_output:
            return False
            
        # Verify the extracted call type is in our list of known types or null
        call_type = parsed_output['call_type']
        return call_type is None or call_type in self.categories
