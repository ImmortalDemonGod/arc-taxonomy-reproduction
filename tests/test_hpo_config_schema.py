"""
Tests for HPO config schema helpers (check_condition, parameter sampling).

These tests verify conditional parameter logic and config schema utilities.
"""
import pytest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hpo.config_schema import check_condition


class TestCheckCondition:
    """Test conditional parameter logic."""
    
    def test_simple_equals_condition_true(self):
        """Test simple equals condition that should pass."""
        condition = {
            "equals": {"encoder_type": "cnn"}
        }
        sampled_params = {"encoder_type": "cnn"}
        
        assert check_condition(condition, sampled_params) is True
    
    def test_simple_equals_condition_false(self):
        """Test simple equals condition that should fail."""
        condition = {
            "equals": {"encoder_type": "cnn"}
        }
        sampled_params = {"encoder_type": "context"}
        
        assert check_condition(condition, sampled_params) is False
    
    def test_multiple_equals_conditions_all_true(self):
        """Test multiple equals conditions where all match."""
        condition = {
            "equals": {
                "encoder_type": "cnn",
                "use_scheduler": True
            }
        }
        sampled_params = {
            "encoder_type": "cnn",
            "use_scheduler": True
        }
        
        assert check_condition(condition, sampled_params) is True
    
    def test_multiple_equals_conditions_one_false(self):
        """Test multiple equals conditions where one doesn't match."""
        condition = {
            "equals": {
                "encoder_type": "cnn",
                "use_scheduler": True
            }
        }
        sampled_params = {
            "encoder_type": "cnn",
            "use_scheduler": False  # This doesn't match
        }
        
        assert check_condition(condition, sampled_params) is False
    
    def test_missing_parameter_returns_false(self):
        """Test that missing parameter in sampled_params returns False."""
        condition = {
            "equals": {"encoder_type": "cnn"}
        }
        sampled_params = {}  # Missing encoder_type
        
        assert check_condition(condition, sampled_params) is False
    
    def test_integer_equality(self):
        """Test equals condition with integer values."""
        condition = {
            "equals": {"num_layers": 3}
        }
        sampled_params = {"num_layers": 3}
        
        assert check_condition(condition, sampled_params) is True
    
    def test_float_equality(self):
        """Test equals condition with float values."""
        condition = {
            "equals": {"lr": 0.001}
        }
        sampled_params = {"lr": 0.001}
        
        assert check_condition(condition, sampled_params) is True
    
    def test_boolean_equality(self):
        """Test equals condition with boolean values."""
        condition = {
            "equals": {"use_scheduler": True}
        }
        sampled_params = {"use_scheduler": True}
        
        assert check_condition(condition, sampled_params) is True
        
        # Test False case
        condition_false = {
            "equals": {"use_scheduler": False}
        }
        assert check_condition(condition_false, sampled_params) is False
    
    def test_string_equality_case_sensitive(self):
        """Test that string equality is case-sensitive."""
        condition = {
            "equals": {"encoder_type": "CNN"}
        }
        sampled_params = {"encoder_type": "cnn"}
        
        assert check_condition(condition, sampled_params) is False
    
    def test_empty_condition(self):
        """Test handling of empty condition dict."""
        condition = {"equals": {}}
        sampled_params = {"encoder_type": "cnn"}
        
        # Empty equals should be True (no constraints to violate)
        assert check_condition(condition, sampled_params) is True
    
    def test_complex_nested_parameters(self):
        """Test with complex parameter names."""
        condition = {
            "equals": {"context_num_heads": 8}
        }
        sampled_params = {
            "encoder_type": "context",
            "context_num_heads": 8,
            "context_num_layers": 4
        }
        
        assert check_condition(condition, sampled_params) is True


class TestConfigSchemaEdgeCases:
    """Test edge cases in config schema handling."""
    
    def test_malformed_condition_missing_equals(self):
        """Test handling of malformed condition without 'equals' key."""
        condition = {"some_other_key": {"param": "value"}}
        sampled_params = {"param": "value"}
        
        # check_condition returns True if no "equals" key is present
        # This is by design - unknown keys are ignored, no constraint = pass
        result = check_condition(condition, sampled_params)
        assert result is True
    
    def test_none_sampled_params(self):
        """Test handling of None sampled_params."""
        condition = {"equals": {"encoder_type": "cnn"}}
        
        try:
            result = check_condition(condition, None)
            assert result is False
        except (TypeError, AttributeError):
            # Expected - can't check condition on None
            pass
    
    def test_none_values_in_equals(self):
        """Test equals condition with None values."""
        condition = {"equals": {"optional_param": None}}
        sampled_params = {"optional_param": None}
        
        assert check_condition(condition, sampled_params) is True
    
    def test_numeric_string_mismatch(self):
        """Test that numeric types and strings don't match."""
        condition = {"equals": {"batch_size": "16"}}
        sampled_params = {"batch_size": 16}
        
        # String "16" should not equal integer 16
        assert check_condition(condition, sampled_params) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
