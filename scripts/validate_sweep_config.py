#!/usr/bin/env python3
"""
Validate HPO sweep configuration for syntax, logic, and compatibility.

Usage:
    python scripts/validate_sweep_config.py configs/hpo/visual_classifier_sweep_v3_intelligent.yaml
"""

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, List


def validate_structure(config: Dict[str, Any]) -> List[str]:
    """Validate basic config structure."""
    errors = []
    required_top_level = ['study_name', 'storage_url', 'n_trials', 'direction', 'fixed', 'param_ranges']
    
    for key in required_top_level:
        if key not in config:
            errors.append(f"Missing required top-level key: '{key}'")
    
    return errors


def validate_param_ranges(param_ranges: Dict[str, Any]) -> List[str]:
    """Validate parameter range definitions."""
    errors = []
    
    for param, spec in param_ranges.items():
        if not isinstance(spec, dict):
            errors.append(f"Parameter '{param}': spec must be a dict, got {type(spec)}")
            continue
        
        if 'type' not in spec:
            errors.append(f"Parameter '{param}': missing 'type' field")
            continue
        
        param_type = spec['type']
        
        # Validate based on type
        if param_type == 'float':
            if 'low' not in spec or 'high' not in spec:
                errors.append(f"Parameter '{param}': float type requires 'low' and 'high'")
            else:
                try:
                    low_val = float(spec['low'])
                    high_val = float(spec['high'])
                    if low_val >= high_val:
                        errors.append(f"Parameter '{param}': low ({low_val}) must be < high ({high_val})")
                except (ValueError, TypeError) as e:
                    errors.append(f"Parameter '{param}': invalid float bounds - {e}")
        
        elif param_type == 'int':
            if 'low' not in spec or 'high' not in spec:
                errors.append(f"Parameter '{param}': int type requires 'low' and 'high'")
            elif spec['low'] >= spec['high']:
                errors.append(f"Parameter '{param}': low ({spec['low']}) must be < high ({spec['high']})")
            elif spec['low'] != int(spec['low']) or spec['high'] != int(spec['high']):
                errors.append(f"Parameter '{param}': int bounds must be integers")
        
        elif param_type == 'categorical':
            if 'choices' not in spec:
                errors.append(f"Parameter '{param}': categorical type requires 'choices'")
            elif not isinstance(spec['choices'], list):
                errors.append(f"Parameter '{param}': choices must be a list")
            elif len(spec['choices']) == 0:
                errors.append(f"Parameter '{param}': choices list is empty")
        
        else:
            errors.append(f"Parameter '{param}': unknown type '{param_type}'")
        
        # Validate conditional logic if present
        if 'condition' in spec:
            cond = spec['condition']
            if 'equals' not in cond:
                errors.append(f"Parameter '{param}': condition must have 'equals' field")
            elif not isinstance(cond['equals'], dict):
                errors.append(f"Parameter '{param}': condition.equals must be a dict")
    
    return errors


def validate_conditional_consistency(param_ranges: Dict[str, Any]) -> List[str]:
    """Validate that conditional parameters reference valid parent parameters."""
    errors = []
    all_params = set(param_ranges.keys())
    
    for param, spec in param_ranges.items():
        if 'condition' in spec:
            cond = spec['condition']
            if 'equals' in cond:
                for parent_param, parent_value in cond['equals'].items():
                    if parent_param not in all_params:
                        errors.append(
                            f"Parameter '{param}': condition references unknown parameter '{parent_param}'"
                        )
                    else:
                        # Check if parent is categorical
                        parent_spec = param_ranges[parent_param]
                        if parent_spec.get('type') != 'categorical':
                            errors.append(
                                f"Parameter '{param}': condition references non-categorical parameter '{parent_param}'"
                            )
                        elif 'choices' in parent_spec and parent_value not in parent_spec['choices']:
                            errors.append(
                                f"Parameter '{param}': condition value '{parent_value}' not in parent choices {parent_spec['choices']}"
                            )
    
    return errors


def check_warnings(config: Dict[str, Any]) -> List[str]:
    """Check for potential issues (not errors, but worth noting)."""
    warnings = []
    
    # Check trial count
    n_trials = config.get('n_trials', 0)
    n_params = len(config.get('param_ranges', {}))
    
    if n_trials < n_params * 3:
        warnings.append(
            f"Low trials/parameter ratio: {n_trials} trials for {n_params} parameters "
            f"(recommended: >{n_params * 5})"
        )
    
    # Check for very narrow ranges
    param_ranges = config.get('param_ranges', {})
    for param, spec in param_ranges.items():
        if spec.get('type') == 'float':
            try:
                low, high = float(spec['low']), float(spec['high'])
                if high / low < 1.5 and spec.get('log', False):
                    warnings.append(
                        f"Parameter '{param}': very narrow log range ({low:.2e} to {high:.2e})"
                    )
            except (ValueError, TypeError, ZeroDivisionError):
                pass  # Skip if can't parse
        elif spec.get('type') == 'categorical':
            choices = spec.get('choices', [])
            if len(choices) == 1:
                warnings.append(
                    f"Parameter '{param}': only 1 choice {choices} (consider moving to 'fixed')"
                )
    
    # Check for degenerate ranges
    for param, spec in param_ranges.items():
        if spec.get('type') == 'int':
            if spec['high'] - spec['low'] < 2:
                warnings.append(
                    f"Parameter '{param}': very narrow int range [{spec['low']}, {spec['high']}] "
                    "(only 1-2 possible values)"
                )
    
    return warnings


def estimate_search_space_size(config: Dict[str, Any]) -> int:
    """Rough estimate of search space size (combinatorial explosion check)."""
    param_ranges = config.get('param_ranges', {})
    size = 1
    
    for param, spec in param_ranges.items():
        param_type = spec.get('type')
        
        if param_type == 'categorical':
            size *= len(spec.get('choices', []))
        elif param_type == 'int':
            try:
                size *= (int(spec.get('high', 0)) - int(spec.get('low', 0)) + 1)
            except (ValueError, TypeError):
                size *= 5  # Default estimate
        elif param_type == 'float':
            # Rough discretization estimate (assume 10 samples per float)
            size *= 10
    
    return size


def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_sweep_config.py <config_file.yaml>")
        sys.exit(1)
    
    config_path = Path(sys.argv[1])
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    print(f"🔍 Validating: {config_path}")
    print("=" * 80)
    
    # Load config
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"❌ YAML parsing error:\n{e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading config:\n{e}")
        sys.exit(1)
    
    all_errors = []
    all_warnings = []
    
    # Run validations
    print("\n📋 Checking structure...")
    errors = validate_structure(config)
    all_errors.extend(errors)
    if errors:
        for err in errors:
            print(f"  ❌ {err}")
    else:
        print("  ✅ Structure valid")
    
    print("\n📊 Checking parameter ranges...")
    errors = validate_param_ranges(config.get('param_ranges', {}))
    all_errors.extend(errors)
    if errors:
        for err in errors:
            print(f"  ❌ {err}")
    else:
        print("  ✅ Parameter ranges valid")
    
    print("\n🔗 Checking conditional consistency...")
    errors = validate_conditional_consistency(config.get('param_ranges', {}))
    all_errors.extend(errors)
    if errors:
        for err in errors:
            print(f"  ❌ {err}")
    else:
        print("  ✅ Conditional parameters consistent")
    
    print("\n⚠️  Checking for warnings...")
    warnings = check_warnings(config)
    all_warnings.extend(warnings)
    if warnings:
        for warn in warnings:
            print(f"  ⚠️  {warn}")
    else:
        print("  ✅ No warnings")
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Study name: {config.get('study_name', 'N/A')}")
    print(f"Trials: {config.get('n_trials', 'N/A')}")
    print(f"Direction: {config.get('direction', 'N/A')}")
    print(f"Fixed parameters: {len(config.get('fixed', {}))}")
    print(f"Searchable parameters: {len(config.get('param_ranges', {}))}")
    
    space_size = estimate_search_space_size(config)
    print(f"Estimated search space: ~{space_size:,} combinations")
    
    if config.get('n_trials'):
        coverage = (config['n_trials'] / space_size) * 100 if space_size > 0 else 0
        print(f"Search space coverage: ~{coverage:.2e}%")
    
    print("\n" + "=" * 80)
    
    if all_errors:
        print(f"❌ VALIDATION FAILED: {len(all_errors)} error(s) found")
        sys.exit(1)
    elif all_warnings:
        print(f"⚠️  VALIDATION PASSED with {len(all_warnings)} warning(s)")
        sys.exit(0)
    else:
        print("✅ VALIDATION PASSED: Config is valid and ready for launch!")
        sys.exit(0)


if __name__ == "__main__":
    main()
