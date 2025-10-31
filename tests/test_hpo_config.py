"""
Test HPO configuration files for validity.

This test catches configuration errors that would only be discovered at runtime,
such as invalid log distributions with low=0.
"""
import pytest
import yaml
from pathlib import Path


def test_sweep_configs_exist():
    """Test that sweep configuration files exist."""
    repo_root = Path(__file__).parent.parent
    main_sweep = repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    test_sweep = repo_root / "configs" / "hpo" / "test_sweep.yaml"
    
    assert main_sweep.exists(), f"Main sweep config not found: {main_sweep}"
    assert test_sweep.exists(), f"Test sweep config not found: {test_sweep}"


def test_log_distributions_have_positive_low():
    """Test that log distributions have low > 0 (log(0) is undefined)."""
    repo_root = Path(__file__).parent.parent
    configs = [
        repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml",
        repo_root / "configs" / "hpo" / "test_sweep.yaml",
    ]
    
    for config_path in configs:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        param_ranges = config.get("param_ranges", {})
        for param_name, param_config in param_ranges.items():
            if param_config.get("type") == "float" and param_config.get("log", False):
                low = param_config.get("low", 0)
                low = float(low)  # Handle scientific notation strings
                assert low > 0, (
                    f"Config {config_path.name}, parameter '{param_name}': "
                    f"log distribution requires low > 0, got low={low}"
                )


def test_required_config_keys():
    """Test that required configuration keys are present."""
    repo_root = Path(__file__).parent.parent
    main_sweep = repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    
    with open(main_sweep) as f:
        config = yaml.safe_load(f)
    
    # Required top-level keys
    required_keys = ["study_name", "storage_url", "n_trials", "direction", "fixed", "param_ranges"]
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"
    
    # Required fixed parameters
    required_fixed = ["data_dir", "labels", "centroids", "epochs", "seed"]
    for key in required_fixed:
        assert key in config["fixed"], f"Missing required fixed parameter: {key}"


def test_conditional_parameters_have_valid_conditions():
    """Test that conditional parameters have properly formatted conditions."""
    repo_root = Path(__file__).parent.parent
    main_sweep = repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    
    with open(main_sweep) as f:
        config = yaml.safe_load(f)
    
    param_ranges = config.get("param_ranges", {})
    for param_name, param_config in param_ranges.items():
        if "condition" in param_config:
            condition = param_config["condition"]
            assert "equals" in condition, (
                f"Parameter '{param_name}' has malformed condition: {condition}"
            )
            assert isinstance(condition["equals"], dict), (
                f"Parameter '{param_name}' condition.equals must be a dict"
            )


def test_pruner_config_valid():
    """Test that pruner configuration is valid."""
    repo_root = Path(__file__).parent.parent
    main_sweep = repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    
    with open(main_sweep) as f:
        config = yaml.safe_load(f)
    
    pruner = config.get("pruner", {})
    assert pruner.get("type") in ["hyperband", "median", None], (
        f"Invalid pruner type: {pruner.get('type')}"
    )
    
    if pruner.get("type") == "hyperband":
        assert "min_resource" in pruner, "Hyperband pruner requires min_resource"
        assert "max_resource" in pruner, "Hyperband pruner requires max_resource"
        assert pruner["min_resource"] < pruner["max_resource"], (
            "min_resource must be < max_resource"
        )


def test_parameter_ranges_valid():
    """Test that parameter ranges are sensible."""
    repo_root = Path(__file__).parent.parent
    main_sweep = repo_root / "configs" / "hpo" / "visual_classifier_sweep.yaml"
    
    with open(main_sweep) as f:
        config = yaml.safe_load(f)
    
    param_ranges = config.get("param_ranges", {})
    for param_name, param_config in param_ranges.items():
        param_type = param_config.get("type")
        
        if param_type in ["int", "float"]:
            low = param_config.get("low")
            high = param_config.get("high")
            assert low is not None, f"Parameter '{param_name}' missing 'low'"
            assert high is not None, f"Parameter '{param_name}' missing 'high'"
            # Convert to float/int to handle scientific notation strings
            if param_type == "float":
                low, high = float(low), float(high)
            else:
                low, high = int(low), int(high)
            assert low < high, f"Parameter '{param_name}': low must be < high"
        
        elif param_type == "categorical":
            choices = param_config.get("choices")
            assert choices, f"Parameter '{param_name}' missing 'choices'"
            assert len(choices) > 0, f"Parameter '{param_name}' has empty choices"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
