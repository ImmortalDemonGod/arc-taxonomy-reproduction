"""
Configuration schema helpers for HPO.

These dataclasses allow Hydra to instantiate Optuna suggestion types from YAML,
enabling declarative search space definitions.
"""
from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict


@dataclass
class IntRange:
    """Integer hyperparameter range for Optuna."""
    low: int
    high: int
    step: int = 1
    log: bool = False
    condition: Optional[Dict[str, Any]] = None  # For conditional parameters
    
    def suggest(self, trial, name: str) -> int:
        """Call trial.suggest_int with this range."""
        return trial.suggest_int(name, self.low, self.high, step=self.step, log=self.log)


@dataclass
class FloatRange:
    """Float hyperparameter range for Optuna."""
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False
    condition: Optional[Dict[str, Any]] = None  # For conditional parameters
    
    def suggest(self, trial, name: str) -> float:
        """Call trial.suggest_float with this range."""
        return trial.suggest_float(name, self.low, self.high, step=self.step, log=self.log)


@dataclass
class CategoricalChoice:
    """Categorical hyperparameter choices for Optuna."""
    choices: List[Any]
    condition: Optional[Dict[str, Any]] = None  # For conditional parameters
    
    def suggest(self, trial, name: str) -> Any:
        """Call trial.suggest_categorical with these choices."""
        return trial.suggest_categorical(name, self.choices)


def check_condition(condition: Optional[Dict[str, Any]], current_params: Dict[str, Any]) -> bool:
    """
    Check if a conditional parameter should be sampled.
    
    Args:
        condition: Condition dict like {"equals": {"encoder_type": "cnn"}}
        current_params: Current trial parameters
    
    Returns:
        True if condition is satisfied or no condition present
    """
    if condition is None:
        return True
    
    if "equals" in condition:
        for key, expected_value in condition["equals"].items():
            if current_params.get(key) != expected_value:
                return False
    
    return True
