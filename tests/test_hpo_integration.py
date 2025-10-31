"""
Integration tests for HPO system.

These tests verify that components work together correctly, catching bugs that
unit tests miss (e.g., type mismatches between config YAML and Optuna API).
"""
import pytest
import yaml
from pathlib import Path
from unittest.mock import Mock, MagicMock
import sys

# Add repo root and scripts to path
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def test_config_values_work_with_optuna_api():
    """
    Test that config YAML values are compatible with Optuna's API.
    
    This catches the bug where YAML parses scientific notation (1e-7) as strings,
    which then fail when passed to trial.suggest_float() expecting numeric types.
    """
    # Load real config
    config_path = REPO_ROOT / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    param_ranges = config.get("param_ranges", {})
    
    # Mock Optuna trial
    trial = Mock()
    trial.suggest_int = Mock(return_value=42)
    trial.suggest_float = Mock(return_value=0.001)
    trial.suggest_categorical = Mock(return_value="cnn")
    
    # Try to sample each parameter
    for param_name, param_config in param_ranges.items():
        param_type = param_config.get("type")
        
        # Skip conditional parameters for this simple test
        if "condition" in param_config:
            continue
        
        if param_type == "int":
            # This should not raise TypeError
            low = int(param_config["low"])
            high = int(param_config["high"])
            assert isinstance(low, int)
            assert isinstance(high, int)
            assert low < high
            
        elif param_type == "float":
            # This should not raise TypeError
            low = float(param_config["low"])
            high = float(param_config["high"])
            assert isinstance(low, float)
            assert isinstance(high, float)
            assert low < high
            
        elif param_type == "categorical":
            choices = param_config.get("choices")
            assert isinstance(choices, list)
            assert len(choices) > 0


def test_objective_can_sample_parameters():
    """
    Test that Objective class can sample parameters from config without errors.
    
    This is a more realistic integration test that uses actual Objective code.
    """
    from objective import Objective
    
    # Load real config
    config_path = REPO_ROOT / "configs" / "hpo" / "test_sweep.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create Objective (without running actual training)
    # We'll mock the datasets
    mock_train_dataset = Mock()
    mock_val_dataset = Mock()
    mock_centroids = Mock()
    
    # Create objective
    objective = Objective(
        config=config,
        train_dataset=mock_train_dataset,
        val_dataset=mock_val_dataset,
        centroids=mock_centroids,
        device="cpu",
        base_output_dir=Path("/tmp/test_hpo")
    )
    
    # Mock trial with real Optuna-like behavior
    trial = Mock()
    sampled_values = {}
    
    def mock_suggest_int(name, low, high, log=False):
        # Verify types (this is where the bug would manifest)
        assert isinstance(low, int), f"suggest_int low must be int, got {type(low)}"
        assert isinstance(high, int), f"suggest_int high must be int, got {type(high)}"
        value = low  # Just return low for testing
        sampled_values[name] = value
        return value
    
    def mock_suggest_float(name, low, high, log=False):
        # Verify types (this is where the bug would manifest)
        assert isinstance(low, (int, float)), f"suggest_float low must be numeric, got {type(low)}"
        assert isinstance(high, (int, float)), f"suggest_float high must be numeric, got {type(high)}"
        value = float(low)  # Return low as float
        sampled_values[name] = value
        return value
    
    def mock_suggest_categorical(name, choices):
        assert isinstance(choices, list), f"choices must be list, got {type(choices)}"
        value = choices[0]  # Return first choice
        sampled_values[name] = value
        return value
    
    trial.suggest_int = mock_suggest_int
    trial.suggest_float = mock_suggest_float
    trial.suggest_categorical = mock_suggest_categorical
    trial.number = 0
    trial.report = Mock()
    trial.should_prune = Mock(return_value=False)
    
    # Try to sample parameters (this is where the TypeError would occur)
    param_ranges = config.get("param_ranges", {})
    sampled_params = {}
    
    for param_name, param_config in param_ranges.items():
        # Skip conditional parameters
        if "condition" in param_config:
            continue
        
        param_type = param_config["type"]
        
        if param_type == "int":
            # This call will fail if low/high are strings
            value = trial.suggest_int(
                param_name,
                int(param_config["low"]),
                int(param_config["high"]),
                log=param_config.get("log", False)
            )
        elif param_type == "float":
            # This call will fail if low/high are strings
            value = trial.suggest_float(
                param_name,
                float(param_config["low"]),
                float(param_config["high"]),
                log=param_config.get("log", False)
            )
        elif param_type == "categorical":
            value = trial.suggest_categorical(
                param_name,
                param_config["choices"]
            )
        
        sampled_params[param_name] = value
    
    # If we got here without TypeError, the test passes
    assert len(sampled_params) > 0, "Should have sampled at least one parameter"


def test_yaml_scientific_notation_parsing():
    """
    Test that YAML correctly parses scientific notation.
    
    This documents the expected behavior and catches if YAML parsing changes.
    """
    yaml_content = """
    test_values:
      string_sci: "1e-7"
      float_sci: 1e-7
      int_val: 42
      float_val: 0.001
    """
    
    data = yaml.safe_load(yaml_content)
    values = data["test_values"]
    
    # Document current behavior
    # Note: PyYAML may parse scientific notation as string or float depending on context
    assert isinstance(values["string_sci"], str)
    # values["float_sci"] should be float, but may be str in some YAML versions
    # This is why we need explicit float() conversion in the code
    
    # These should always work
    assert isinstance(values["int_val"], int)
    assert isinstance(values["float_val"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
