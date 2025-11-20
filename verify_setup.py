#!/usr/bin/env python3
"""
Reproduction Package Setup Verification

Verifies that all critical files and dependencies are present for reproducing
the paper's empirical claims.

Usage:
    python verify_setup.py
"""

from pathlib import Path
import sys


def print_check(name: str, passed: bool, detail: str = "") -> bool:
    """Print a check result."""
    status = "✅" if passed else "❌"
    print(f"{status} {name}", end="")
    if detail:
        print(f" - {detail}")
    else:
        print()
    return passed


def check_critical_data_files():
    """Check that critical data files exist for reproduction."""
    print("\n🔍 Checking Critical Data Files...")
    
    required_files = {
        "data/taxonomy/all_tasks_classified.json": "Master taxonomy (400 tasks)",
        "data/taxonomy/tasks_by_category.json": "Category mappings",
        "data/taxonomy/s3_final_classification.json": "S3 heterogeneity data",
        "data/external_validation/vitarc_appendix_tables.csv": "ViTARC validation data",
        "outputs/atomic_lora_training_summary.json": "LoRA training results (§7.1, §7.2)",
        "external/re-arc/generators.py": "re-arc generator (§5.2)"
    }
    
    all_present = True
    missing = []
    
    for filepath, description in required_files.items():
        path = Path(filepath)
        exists = path.exists()
        all_present &= print_check(
            description,
            exists,
            f"({filepath})" if exists else f"MISSING: {filepath}"
        )
        if not exists:
            missing.append(filepath)
    
    if "external/re-arc/generators.py" in missing:
        print("\n   💡 Run: git submodule update --init --recursive")
    
    return all_present


def check_reproduction_scripts():
    """Check that all 6 key reproduction scripts exist."""
    print("\n📜 Checking Reproduction Scripts...")
    
    scripts = {
        "scripts/0_generate_taxonomy_classification.py": "§5.2 Taxonomy Validation",
        "scripts/calculate_compositional_gap.py": "§7.1 Compositional Gap",
        "scripts/verify_a2_failures.py": "§7.2 A2 Ceiling",
        "scripts/arc_agi_2_analysis.py": "§7.3 ARC-AGI-2",
        "scripts/0_analyze_s3_heterogeneity.py": "§7.4 S3 Heterogeneity",
        "scripts/analyze_vitarc_performance.py": "§7.5 ViTARC Validation"
    }
    
    all_present = True
    for filepath, description in scripts.items():
        path = Path(filepath)
        all_present &= print_check(description, path.exists(), f"({filepath})")
    
    return all_present


def check_figure_scripts():
    """Check that figure generation scripts exist (may be in top-level scripts/)."""
    print("\n📊 Checking Figure Generation Scripts...")
    
    # These may be in either reproduction/scripts/ or top-level ../scripts/
    figure_scripts = {
        "generate_smoking_gun_figure.py": "§7.2 A2 Smoking Gun",
        "generate_arc_agi_2_comparison.py": "§7.3 ARC-AGI-2 Comparison",
        "generate_failure_concentration.py": "§7.3 Failure Concentration",
        "generate_s3_performance_profiles.py": "§7.4 S3 Performance",
        "generate_compositional_gap_sensitivity.py": "Gap Sensitivity"
    }
    
    all_present = True
    for script_name, description in figure_scripts.items():
        # Check both locations
        local_path = Path("scripts") / script_name
        parent_path = Path("..") / "scripts" / script_name
        
        if local_path.exists():
            all_present &= print_check(description, True, f"(scripts/{script_name})")
        elif parent_path.exists():
            all_present &= print_check(description, True, f"(../scripts/{script_name})")
        else:
            all_present &= print_check(description, False, f"MISSING: {script_name}")
    
    return all_present


def check_python_environment():
    """Check Python version and key packages."""
    print("\n🐍 Checking Python Environment...")
    
    # Check Python version
    version = sys.version_info
    python_ok = version.major == 3 and version.minor >= 8
    print_check(
        f"Python {version.major}.{version.minor}.{version.micro}",
        python_ok,
        "OK" if python_ok else "Need Python 3.8+"
    )
    
    # Check key packages
    packages = ["torch", "numpy", "pandas", "scipy", "yaml"]
    for pkg in packages:
        try:
            __import__(pkg)
            print_check(pkg, True, "installed")
        except ImportError:
            print_check(pkg, False, "MISSING - run: pip install -r requirements.txt")
    
    return python_ok


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("🔬 Neural Affinity Framework - Reproduction Package Verification")
    print("=" * 70)
    
    results = []
    results.append(("Python Environment", check_python_environment()))
    results.append(("Critical Data Files", check_critical_data_files()))
    results.append(("Reproduction Scripts", check_reproduction_scripts()))
    results.append(("Figure Scripts", check_figure_scripts()))
    
    print("\n" + "=" * 70)
    print("📊 Summary")
    print("=" * 70)
    
    all_passed = all(passed for _, passed in results)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:10} {name}")
    
    print("=" * 70)
    if all_passed:
        print("✅ Setup verification PASSED - Ready to reproduce!")
        print("\nNext steps:")
        print("  1. python scripts/calculate_compositional_gap.py")
        print("  2. python scripts/verify_a2_failures.py")
        print("  3. See README.md for complete reproduction guide")
        return 0
    else:
        print("❌ Setup verification FAILED - Fix issues above")
        print("\nCommon fixes:")
        print("  - pip install -r requirements.txt")
        print("  - git submodule update --init --recursive")
        return 1


if __name__ == "__main__":
    sys.exit(main())
