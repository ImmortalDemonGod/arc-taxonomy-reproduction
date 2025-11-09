#!/bin/bash
# Verification script for ablation redesign setup
# Run from: cultivation/systems/arc_reactor/publications/arc_taxonomy_2025/reproduction

set -e

echo "=================================="
echo "Ablation Redesign: Setup Verification"
echo "=================================="
echo ""

cd "$(dirname "$0")"

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check existing scripts
echo "1. Checking existing training scripts..."
echo ""

if [ -f "scripts/train_exp0_encoder_decoder.py" ]; then
    echo -e "${GREEN}✓${NC} scripts/train_exp0_encoder_decoder.py exists"
else
    echo -e "${RED}✗${NC} scripts/train_exp0_encoder_decoder.py MISSING"
fi

if [ -f "scripts/train_exp1_grid2d_pe.py" ]; then
    echo -e "${GREEN}✓${NC} scripts/train_exp1_grid2d_pe.py exists"
else
    echo -e "${RED}✗${NC} scripts/train_exp1_grid2d_pe.py MISSING"
fi

if [ -f "scripts/train_exp3_champion.py" ]; then
    echo -e "${GREEN}✓${NC} scripts/train_exp3_champion.py exists"
else
    echo -e "${RED}✗${NC} scripts/train_exp3_champion.py MISSING"
fi

echo ""

# 2. Check max_grid_size in Champion script
echo "2. Verifying max_grid_size=30 in Champion..."
echo ""

MAX_GRID_COUNT=$(grep -c "max_grid_size=30" scripts/train_exp3_champion.py || echo "0")
if [ "$MAX_GRID_COUNT" -ge "2" ]; then
    echo -e "${GREEN}✓${NC} Champion uses max_grid_size=30 (found $MAX_GRID_COUNT occurrences)"
else
    echo -e "${RED}✗${NC} Champion max_grid_size setting unclear"
fi

echo ""

# 3. Check hyperparameters match
echo "3. Checking hyperparameters consistency..."
echo ""

echo "Learning rate in all scripts:"
grep -n "learning_rate=" scripts/train_exp0_encoder_decoder.py | head -1
grep -n "learning_rate=" scripts/train_exp1_grid2d_pe.py | head -1  
grep -n "learning_rate=" scripts/train_exp3_champion.py | head -1

echo ""

# 4. Test Exp0 smoke run
echo "4. Testing Exp0 (E-D Baseline) with fast_dev_run..."
echo ""

if python scripts/train_exp0_encoder_decoder.py --fast_dev_run 1 2>&1 | grep -q "Training:"; then
    echo -e "${GREEN}✓${NC} Exp0 smoke test PASSED"
else
    echo -e "${RED}✗${NC} Exp0 smoke test FAILED"
fi

echo ""

# 5. Test Exp1 smoke run
echo "5. Testing Exp1 (E-D + Grid2D) with fast_dev_run..."
echo ""

if python scripts/train_exp1_grid2d_pe.py --fast_dev_run 1 2>&1 | grep -q "Exp 1:"; then
    echo -e "${GREEN}✓${NC} Exp1 smoke test PASSED"
else
    echo -e "${RED}✗${NC} Exp1 smoke test FAILED"
fi

echo ""

# 6. Test Exp4 smoke run
echo "6. Testing Exp4 (Champion) with fast_dev_run..."
echo ""

if python scripts/train_exp3_champion.py --fast_dev_run 1 2>&1 | grep -q "CHAMPION"; then
    echo -e "${GREEN}✓${NC} Exp4 smoke test PASSED"
else
    echo -e "${RED}✗${NC} Exp4 smoke test FAILED"
fi

echo ""

# 7. Check missing scripts
echo "7. Checking for missing scripts (NEW architecture tests)..."
echo ""

if [ -f "scripts/train_exp2_ed_perminv.py" ]; then
    echo -e "${GREEN}✓${NC} scripts/train_exp2_ed_perminv.py exists"
else
    echo -e "${YELLOW}!${NC} scripts/train_exp2_ed_perminv.py MISSING (needs creation)"
fi

if [ -f "scripts/train_exp3_ed_context.py" ]; then
    echo -e "${GREEN}✓${NC} scripts/train_exp3_ed_context.py exists"
else
    echo -e "${YELLOW}!${NC} scripts/train_exp3_ed_context.py MISSING (needs creation)"
fi

echo ""

# 8. Check missing architectures
echo "8. Checking for missing architecture files..."
echo ""

if [ -f "src/models/ed_with_perminv.py" ]; then
    echo -e "${GREEN}✓${NC} src/models/ed_with_perminv.py exists"
else
    echo -e "${YELLOW}!${NC} src/models/ed_with_perminv.py MISSING (needs creation)"
fi

if [ -f "src/models/ed_with_context.py" ]; then
    echo -e "${GREEN}✓${NC} src/models/ed_with_context.py exists"
else
    echo -e "${YELLOW}!${NC} src/models/ed_with_context.py MISSING (needs creation)"
fi

echo ""
echo "=================================="
echo "Verification Complete"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Create missing architecture files (E-D + PermInv, E-D + Context)"
echo "2. Create missing training scripts"
echo "3. Run Phase 1 validation (10 epochs)"
echo "4. Launch Phase 2 full experiment (200 epochs, 5 seeds)"
