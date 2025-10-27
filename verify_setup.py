#!/usr/bin/env python3
"""
Verify ARC Taxonomy reproduction package is ready for cloud deployment.

Checks:
- Python version
- Dependencies
- GPU availability
- Data files
- Model architectures
- Training scripts
"""
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print section header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_check(name, status, details=""):
    """Print check result."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name:40s} {details}")
    return status


def check_python_version():
    """Check Python version >= 3.10."""
    version = sys.version_info
    required = (3, 10)
    status = version >= required
    details = f"v{version.major}.{version.minor}.{version.micro}"
    return print_check("Python Version", status, details)


def check_dependencies():
    """Check core dependencies are installed."""
    deps = {
        "torch": "PyTorch",
        "pytorch_lightning": "PyTorch Lightning",
        "numpy": "NumPy",
        "omegaconf": "OmegaConf",
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            __import__(module)
            print_check(f"  {name}", True)
        except ImportError:
            print_check(f"  {name}", False, "NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            details = f"{gpu_count}x {gpu_name}"
            return print_check("GPU Availability", True, details)
        else:
            return print_check("GPU Availability", False, "CPU only (slow)")
    except:
        return print_check("GPU Availability", False, "ERROR checking")


def check_data_files():
    """Check data files exist."""
    data_dir = Path("data/tasks")
    if not data_dir.exists():
        return print_check("Data Directory", False, "data/tasks/ not found")
    
    json_files = list(data_dir.glob("*.json"))
    if len(json_files) == 0:
        return print_check("Task Files", False, "No JSON files found")
    
    details = f"{len(json_files)} task files"
    return print_check("Task Files", True, details)


def check_model_imports():
    """Check model imports work."""
    models = [
        ("src.models.decoder_only_lightning", "DecoderOnlyLightningModule"),
        ("src.models.encoder_decoder_lightning", "EncoderDecoderLightningModule"),
        ("src.models.champion_lightning", "ChampionLightningModule"),
    ]
    
    all_ok = True
    for module_path, class_name in models:
        try:
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print_check(f"  {class_name}", True)
        except Exception as e:
            print_check(f"  {class_name}", False, str(e)[:40])
            all_ok = False
    
    return all_ok


def check_training_scripts():
    """Check training scripts exist and are executable."""
    scripts = [
        "scripts/train_decoder_only.py",
        "scripts/train_encoder_decoder.py",
        "scripts/train_champion.py",
        "scripts/test_all_training.py",
    ]
    
    all_ok = True
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print_check(f"  {script_path.name}", True)
        else:
            print_check(f"  {script_path.name}", False, "NOT FOUND")
            all_ok = False
    
    return all_ok


def check_disk_space():
    """Check available disk space."""
    try:
        result = subprocess.run(
            ["df", "-h", "."],
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split('\n')
        if len(lines) > 1:
            parts = lines[1].split()
            available = parts[3]
            details = f"{available} available"
            # Rough check: need at least 2GB
            avail_gb = float(available.replace('G', '').replace('M', '')) if 'G' in available else 0
            status = avail_gb > 2
            return print_check("Disk Space", status, details)
    except:
        pass
    
    return print_check("Disk Space", True, "Unable to check")


def run_quick_test():
    """Run quick training test."""
    try:
        print("\n  Running quick validation test (this may take 30-60 seconds)...")
        result = subprocess.run(
            [sys.executable, "scripts/test_all_training.py"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode == 0 and "ALL MODELS READY" in result.stdout:
            print_check("Training Test", True, "All models passed")
            return True
        else:
            print_check("Training Test", False, "Test failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print_check("Training Test", False, "Timeout")
        return False
    except Exception as e:
        print_check("Training Test", False, str(e)[:40])
        return False


def main():
    """Run all verification checks."""
    print_header("ARC TAXONOMY REPRODUCTION - SETUP VERIFICATION")
    
    # Track overall status
    checks = []
    
    # Python version
    print_header("1. Python Environment")
    checks.append(check_python_version())
    
    # Dependencies
    print_header("2. Dependencies")
    checks.append(check_dependencies())
    
    # GPU
    print_header("3. Hardware")
    checks.append(check_gpu())
    checks.append(check_disk_space())
    
    # Data
    print_header("4. Data Files")
    checks.append(check_data_files())
    
    # Models
    print_header("5. Model Imports")
    checks.append(check_model_imports())
    
    # Scripts
    print_header("6. Training Scripts")
    checks.append(check_training_scripts())
    
    # Quick test
    print_header("7. End-to-End Test")
    checks.append(run_quick_test())
    
    # Summary
    print_header("SUMMARY")
    passed = sum(checks)
    total = len(checks)
    
    if all(checks):
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print("\n🚀 Ready for training!")
        print("\nQuick start:")
        print("  ./run_training.sh champion")
        print("  # or")
        print("  python scripts/train_champion.py")
        return 0
    else:
        print(f"❌ SOME CHECKS FAILED ({passed}/{total} passed)")
        print("\n⚠️  Please fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Add data files: Copy JSON files to data/tasks/")
        print("  - Check GPU drivers: nvidia-smi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
