import unittest
from unittest.mock import MagicMock, patch
import os

# Ensure the test runner can find the RobustEvaluator class
# This might be needed if running tests from a directory other than the root,
# or if the package structure isn't perfectly picked up.
# For `python -m unittest discover libs`, this might not be strictly necessary
# if __init__.py files are correctly placed.
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from langchain_robust_evaluator.evaluator import RobustEvaluator
# Assuming PrimaryParser is a class that might be instantiated by RobustEvaluator if not provided
# from langchain_robust_evaluator.parsers.primary_parser import PrimaryParser


class TestRobustEvaluatorAmplification(unittest.TestCase):

    def setUp(self):
        """Set up a RobustEvaluator instance with mocked dependencies."""
        self.mock_model = MagicMock()
        self.mock_model.invoke.return_value = {"call_type": "MOCKED_RESPONSE"} # Default mock response

        self.mock_primary_parser = MagicMock()
        self.mock_primary_parser.invoke.return_value = {"call_type": "PRIMARY_PARSED", "confidence": 0.9}
        
        self.mock_backup_parser = MagicMock()
        self.mock_backup_parser.invoke.return_value = {"call_type": "BACKUP_PARSED", "confidence": 0.8}
        
        self.mock_negative_checker = MagicMock()
        self.mock_negative_checker.invoke.return_value = {"is_negative": False}

        self.mock_schema_validator = MagicMock()
        self.mock_schema_validator.invoke.return_value = {"is_valid": True, "errors": None} # Pass validation

        self.mock_combiner = MagicMock()
        self.mock_combiner.invoke.return_value = {"call_type": "COMBINED", "confidence": 0.95, "needs_human": False}

        self.categories = ["SUPPORT", "BILLING", "SALES"]
        
        # Mock os.getenv to avoid real API key requirement for ChatAnthropic instantiation
        # if the model itself is not mocked early enough (e.g. if RobustEvaluator initializes it before we can replace)
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_key"}):
            self.evaluator = RobustEvaluator(
                categories=self.categories,
                model=self.mock_model, # Pass the mocked model directly
                parsers={
                    "primary": self.mock_primary_parser,
                    "backup": self.mock_backup_parser,
                    "negative": self.mock_negative_checker
                },
                combiner=self.mock_combiner,
                schema_validator=self.mock_schema_validator,
                prompt_templates={ # Provide templates as amplify method uses them
                    "primary": "Input: {input_text}\nCategories: {categories}"
                }
            )

    def test_amplify_basic_flow_and_instruction_variance(self):
        """Test basic amplification flow, instruction generation, and debug output."""
        input_text = "My internet is down."
        
        # First call with factor 1.5
        amplification_factor_1 = 1.5
        expected_instruction_1 = f"Amplify your response, providing more detail and expressing higher confidence by a factor of {amplification_factor_1}."
        
        # Reset mock for primary parser to check calls specifically for this test run
        self.mock_primary_parser.reset_mock()
        # Define a specific return value for the amplified call if needed for other assertions
        amplified_primary_response_1 = {"call_type": "AMPLIFIED_SUPPORT", "confidence": 0.99}
        
        # Side effect to differentiate between first (original) and second (amplified) call to primary_parser
        # The actual primary parser is called for each parser in the initial loop,
        # then re-invoked for amplification. So, initial_results + amplified_result.
        initial_primary_call_return = {"call_type": "INITIAL_PRIMARY", "confidence": 0.8}
        self.mock_primary_parser.invoke.side_effect = [
            initial_primary_call_return, # For the initial call to primary within the loop
            amplified_primary_response_1 # For the specific amplified call
        ]

        result_1 = self.evaluator.amplify(input_text, amplification_factor=amplification_factor_1)

        self.assertIn("call_type", result_1)
        self.assertIn("confidence", result_1)
        self.assertIn("_debug", result_1)
        self.assertEqual(result_1["_debug"]["amplification_factor"], amplification_factor_1)
        self.assertEqual(result_1["_debug"]["amplification_instruction"], expected_instruction_1)
        self.assertIn("amplified_primary_output", result_1["_debug"])
        self.assertEqual(result_1["_debug"]["amplified_primary_output"], amplified_primary_response_1)

        # Check primary parser calls
        # First call is the initial one in the loop.
        # Second call is the amplified one.
        self.assertEqual(self.mock_primary_parser.invoke.call_count, 2)
        
        # The first call to primary_parser.invoke (in the initial loop)
        self.mock_primary_parser.invoke.assert_any_call(input_text)
        
        # The second call to primary_parser.invoke (the amplified one)
        amplified_call_args = self.mock_primary_parser.invoke.call_args_list[1][0][0]
        self.assertIn(input_text, amplified_call_args)
        self.assertIn(expected_instruction_1, amplified_call_args)
        self.assertIn(", ".join(self.categories), amplified_call_args) # Check categories are in prompt

        # Second call with factor 2.5 for instruction variance
        self.mock_primary_parser.reset_mock() # Reset for the next call
        amplified_primary_response_2 = {"call_type": "SUPER_AMPLIFIED_SUPPORT", "confidence": 1.0}
        self.mock_primary_parser.invoke.side_effect = [
             {"call_type": "INITIAL_PRIMARY_AGAIN", "confidence": 0.81}, # Initial call
             amplified_primary_response_2 # Amplified call
        ]

        amplification_factor_2 = 2.5
        expected_instruction_2 = "Amplify your response significantly, providing extensive details and expressing strong confidence."
        result_2 = self.evaluator.amplify(input_text, amplification_factor=amplification_factor_2)
        
        self.assertEqual(result_2["_debug"]["amplification_factor"], amplification_factor_2)
        self.assertEqual(result_2["_debug"]["amplification_instruction"], expected_instruction_2)
        self.assertEqual(result_2["_debug"]["amplified_primary_output"], amplified_primary_response_2)

    def test_amplify_varying_amplification_factors_instructions(self):
        """Test that different amplification_factor values result in different instructions."""
        input_text = "Need help."
        factors_instructions = {
            0.5: f"Slightly amplify your response, providing a bit more detail. Factor: {0.5}.",
            1.0: "Provide your standard response.",
            1.5: f"Amplify your response, providing more detail and expressing higher confidence by a factor of {1.5}.",
            2.5: "Amplify your response significantly, providing extensive details and expressing strong confidence."
        }

        previous_instructions = set()

        for factor, expected_instruction in factors_instructions.items():
            # Reset relevant mocks for each iteration if they accumulate state
            self.mock_primary_parser.reset_mock()
            # Ensure primary parser returns something for both calls it expects
            self.mock_primary_parser.invoke.side_effect = [
                {"call_type": "INITIAL", "confidence": 0.7},
                {"call_type": f"AMPLIFIED_{factor}", "confidence": 0.9}
            ]
            
            result = self.evaluator.amplify(input_text, amplification_factor=factor)
            instruction_in_debug = result["_debug"]["amplification_instruction"]
            self.assertEqual(instruction_in_debug, expected_instruction)
            self.assertNotIn(instruction_in_debug, previous_instructions, f"Instruction for factor {factor} was not unique or matched a previous one unexpectedly.")
            previous_instructions.add(instruction_in_debug)

    def test_amplify_schema_validator_called_with_amplified_output(self):
        """Test that schema_validator.invoke is called with the amplified primary parser's output."""
        input_text = "Issue with login."
        amplified_primary_response = {"call_type": "AMPLIFIED_ACCOUNT", "confidence": 0.92}
        
        # The primary parser is called multiple times, first for initial, then for amplified.
        # We care about the input to schema_validator, which should be the amplified one.
        self.mock_primary_parser.invoke.side_effect = [
            {"call_type": "INITIAL_PRIMARY", "confidence": 0.8}, # Initial call
            amplified_primary_response  # Amplified call
        ]
        
        self.evaluator.amplify(input_text, amplification_factor=1.5)

        self.mock_schema_validator.invoke.assert_called_once_with(amplified_primary_response)

    def test_amplify_combiner_called_with_amplified_output_in_primary_slot(self):
        """Test that combiner.invoke is called with the amplified result replacing the original primary result."""
        input_text = "My payment failed."
        
        initial_primary_response = {"call_type": "PRIMARY_BILLING_INITIAL", "confidence": 0.85}
        amplified_primary_response = {"call_type": "PRIMARY_BILLING_AMPLIFIED", "confidence": 0.97}
        backup_response = {"call_type": "BACKUP_BILLING", "confidence": 0.8}
        negative_response = {"is_negative": False}

        self.mock_primary_parser.invoke.side_effect = [initial_primary_response, amplified_primary_response]
        self.mock_backup_parser.invoke.return_value = backup_response
        self.mock_negative_checker.invoke.return_value = negative_response
        
        # Expected validation result based on amplified primary
        expected_validation_result = {"is_valid": True, "errors": None} 
        self.mock_schema_validator.invoke.return_value = expected_validation_result

        self.evaluator.amplify(input_text, amplification_factor=1.7)

        self.mock_combiner.invoke.assert_called_once()
        combiner_input_args = self.mock_combiner.invoke.call_args[0][0]
        
        expected_parser_results_for_combiner = {
            "primary": amplified_primary_response, # This is key: primary slot should have amplified result
            "backup": backup_response,
            "negative": negative_response,
            "primary_amplified": amplified_primary_response # This is also present from the internal logic
        }
        
        self.assertEqual(combiner_input_args["parser_results"], expected_parser_results_for_combiner)
        self.assertEqual(combiner_input_args["validation_result"], expected_validation_result)
        self.assertEqual(combiner_input_args["input_text"], input_text)

    def test_amplify_uses_default_primary_prompt_if_not_in_templates(self):
        """Test that amplify constructs a prompt for primary if not in prompt_templates."""
        input_text = "Test input"
        amplification_factor = 1.2
        expected_instruction = f"Amplify your response, providing more detail and expressing higher confidence by a factor of {amplification_factor}."

        # Create an evaluator without primary prompt template
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "fake_key"}):
            evaluator_no_primary_template = RobustEvaluator(
                categories=self.categories,
                model=self.mock_model,
                parsers={
                    "primary": self.mock_primary_parser,
                },
                combiner=self.mock_combiner,
                schema_validator=self.mock_schema_validator,
                prompt_templates={} # Empty prompt_templates
            )
        
        self.mock_primary_parser.reset_mock()
        self.mock_primary_parser.invoke.side_effect = [
            {"call_type": "INITIAL", "confidence": 0.8},
            {"call_type": "AMPLIFIED", "confidence": 0.9}
        ]

        evaluator_no_primary_template.amplify(input_text, amplification_factor=amplification_factor)
        
        self.assertEqual(self.mock_primary_parser.invoke.call_count, 2)
        amplified_call_prompt = self.mock_primary_parser.invoke.call_args_list[1][0][0]

        # Check for default prompt structure elements
        self.assertIn(f"Classify the following text: {input_text}", amplified_call_prompt)
        self.assertIn(f"Categories: {', '.join(self.categories)}", amplified_call_prompt)
        self.assertIn(expected_instruction, amplified_call_prompt)


if __name__ == "__main__":
    unittest.main()
