#!/bin/bash
#
# Quick training launcher for ARC Taxonomy Ablation Study
# Usage: ./run_training.sh [baseline|exp0|exp1|exp2|exp3|all|test]
#
# Experiments:
#   baseline - Decoder-Only (catastrophic failure baseline)
#   exp0     - Generic Encoder-Decoder (+17% gain)
#   exp1     - + Grid2D Positional Encoding (+15% gain)
#   exp2     - + Permutation-Invariant Embedding (+3% gain)
#   exp3     - + Context Bridge (Champion) (+24% gain)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_info "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.10+ required, found $python_version"
    exit 1
fi
print_info "Python version OK: $python_version"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_warn "No virtual environment detected. Recommended to use venv."
    print_info "Run: python3 -m venv venv && source venv/bin/activate"
fi

# Check if dependencies are installed
print_info "Checking dependencies..."
if ! python3 -c "import torch, pytorch_lightning" 2>/dev/null; then
    print_error "Dependencies not installed. Run: pip install -r requirements.txt"
    exit 1
fi
print_info "Dependencies OK"

# Check for GPU
print_info "Checking for GPU..."
if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    print_info "GPU detected: $gpu_name"
else
    print_warn "No GPU detected. Training will use CPU (slow)"
fi

# Check data directory
print_info "Checking data..."
if [ ! -d "data/distributional_alignment" ] || [ -z "$(ls -A data/distributional_alignment/*.json 2>/dev/null)" ]; then
    print_error "No task files found in data/distributional_alignment/"
    print_info "Please ensure data/distributional_alignment/ contains ARC JSON files"
    exit 1
fi
task_count=$(ls -1 data/distributional_alignment/*.json 2>/dev/null | grep -v -E "(split_manifest|generation_statistics|task_categories)" | wc -l)
print_info "Found $task_count task files"

# Determine which experiment to run
experiment="${1:-exp3}"

# Create logs directory if it doesn't exist
mkdir -p logs/console_output

# Generate timestamp for log file
timestamp=$(date +"%Y%m%d_%H%M%S")

case "$experiment" in
    baseline|-1)
        print_info "Training Baseline: Decoder-Only (catastrophic failure)..."
        log_file="logs/console_output/baseline_${timestamp}.log"
        print_info "Console output will be saved to: $log_file"
        python3 scripts/train_baseline_decoder_only.py 2>&1 | tee "$log_file"
        print_info "Baseline training complete! Log: $log_file"
        ;;
    exp0|0)
        print_info "Training Exp 0: Generic Encoder-Decoder (+17% over baseline)..."
        log_file="logs/console_output/exp0_${timestamp}.log"
        print_info "Console output will be saved to: $log_file"
        python3 scripts/train_exp0_encoder_decoder.py 2>&1 | tee "$log_file"
        print_info "Exp 0 training complete! Log: $log_file"
        ;;
    exp1|1)
        print_info "Training Exp 1: + Grid2D Positional Encoding (+15% over Exp 0)..."
        log_file="logs/console_output/exp1_${timestamp}.log"
        print_info "Console output will be saved to: $log_file"
        python3 scripts/train_exp1_grid2d_pe.py 2>&1 | tee "$log_file"
        print_info "Exp 1 training complete! Log: $log_file"
        ;;
    exp2|2)
        print_info "Training Exp 2: + Permutation-Invariant Embedding (+3% over Exp 1)..."
        log_file="logs/console_output/exp2_${timestamp}.log"
        print_info "Console output will be saved to: $log_file"
        python3 scripts/train_exp2_perminv.py 2>&1 | tee "$log_file"
        print_info "Exp 2 training complete! Log: $log_file"
        ;;
    exp3|3|champion)
        print_info "Training Exp 3 (Champion): + Context Bridge (+24% over Exp 2)..."
        log_file="logs/console_output/exp3_champion_${timestamp}.log"
        print_info "Console output will be saved to: $log_file"
        print_info "You can monitor progress with: tail -f $log_file"
        python3 scripts/train_exp3_champion.py 2>&1 | tee "$log_file"
        print_info "Champion training complete! Log: $log_file"
        ;;
    all)
        print_info "Training ALL ablation experiments sequentially..."
        print_info ""
        print_info "[1/5] Baseline (Decoder-Only)..."
        python3 scripts/train_baseline_decoder_only.py 2>&1 | tee "logs/console_output/baseline_${timestamp}.log"
        print_info ""
        print_info "[2/5] Exp 0 (Encoder-Decoder)..."
        python3 scripts/train_exp0_encoder_decoder.py 2>&1 | tee "logs/console_output/exp0_${timestamp}.log"
        print_info ""
        print_info "[3/5] Exp 1 (+ Grid2D PE)..."
        python3 scripts/train_exp1_grid2d_pe.py 2>&1 | tee "logs/console_output/exp1_${timestamp}.log"
        print_info ""
        print_info "[4/5] Exp 2 (+ PermInv)..."
        python3 scripts/train_exp2_perminv.py 2>&1 | tee "logs/console_output/exp2_${timestamp}.log"
        print_info ""
        print_info "[5/5] Exp 3 (Champion)..."
        python3 scripts/train_exp3_champion.py 2>&1 | tee "logs/console_output/exp3_${timestamp}.log"
        print_info ""
        print_info "All 5 ablation experiments complete!"
        print_info "Logs saved to: logs/console_output/*_${timestamp}.log"
        ;;
    test)
        print_info "Running comprehensive ablation test..."
        python3 scripts/test_complete_ablation.py
        ;;
    *)
        print_error "Unknown experiment: $experiment"
        echo "Usage: $0 [baseline|exp0|exp1|exp2|exp3|all|test]"
        echo ""
        echo "Ablation Study Experiments:"
        echo "  baseline  - Decoder-Only (catastrophic failure baseline)"
        echo "  exp0      - Generic Encoder-Decoder (+17% gain)"
        echo "  exp1      - + Grid2D Positional Encoding (+15% gain)"
        echo "  exp2      - + Permutation-Invariant Embedding (+3% gain)"
        echo "  exp3      - + Context Bridge (Champion) (+24% gain) [default]"
        echo "  all       - Train all 5 experiments sequentially"
        echo "  test      - Run quick validation test"
        echo ""
        echo "Aliases:"
        echo "  -1, 0, 1, 2, 3   - Numeric experiment IDs"
        echo "  champion         - Alias for exp3"
        exit 1
        ;;
esac

print_info "Done!"
