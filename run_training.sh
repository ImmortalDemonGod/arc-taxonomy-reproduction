#!/bin/bash
#
# Quick training launcher for ARC Taxonomy experiments
# Usage: ./run_training.sh [decoder-only|encoder-decoder|champion|all]
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
experiment="${1:-champion}"

case "$experiment" in
    decoder-only|-1)
        print_info "Training Decoder-Only baseline (Exp -1)..."
        python3 scripts/train_decoder_only.py
        ;;
    encoder-decoder|0)
        print_info "Training Encoder-Decoder baseline (Exp 0)..."
        python3 scripts/train_encoder_decoder.py
        ;;
    champion|3)
        print_info "Training Champion model (Exp 3)..."
        
        # Create logs directory if it doesn't exist
        mkdir -p logs/console_output
        
        # Generate timestamp for log file
        timestamp=$(date +"%Y%m%d_%H%M%S")
        log_file="logs/console_output/champion_training_${timestamp}.log"
        
        print_info "Console output will be saved to: $log_file"
        print_info "You can monitor progress with: tail -f $log_file"
        
        # Run training with output both to console AND log file
        python3 scripts/train_champion.py 2>&1 | tee "$log_file"
        
        print_info "Training complete! Full log saved to: $log_file"
        ;;
    all)
        print_info "Training ALL experiments sequentially..."
        print_info "[1/3] Decoder-Only..."
        python3 scripts/train_decoder_only.py
        print_info "[2/3] Encoder-Decoder..."
        python3 scripts/train_encoder_decoder.py
        print_info "[3/3] Champion..."
        python3 scripts/train_champion.py
        print_info "All experiments complete!"
        ;;
    test)
        print_info "Running quick test..."
        python3 scripts/test_all_training.py
        ;;
    *)
        print_error "Unknown experiment: $experiment"
        echo "Usage: $0 [decoder-only|encoder-decoder|champion|all|test]"
        echo ""
        echo "Options:"
        echo "  decoder-only      - Train Decoder-Only baseline (Exp -1)"
        echo "  encoder-decoder   - Train Encoder-Decoder baseline (Exp 0)"
        echo "  champion          - Train Champion model (Exp 3) [default]"
        echo "  all               - Train all experiments sequentially"
        echo "  test              - Run quick validation test"
        exit 1
        ;;
esac

print_info "Done!"
