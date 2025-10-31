#!/bin/bash
#
# Comprehensive HPO Sweep Analysis Script
# Systematically analyzes the visual_classifier_cnn_vs_context_v2_expanded study
# using all available inspection tools from inspect_optuna_db.py
#

set -e  # Exit on error

# Configuration
STORAGE_URL="***REMOVED***"
STUDY_NAME="visual_classifier_cnn_vs_context_v2_expanded"
ANALYSIS_DIR="outputs/visual_classifier/hpo/analysis_$(date +%Y%m%d_%H%M%S)"
INSPECT_SCRIPT="../../../jarc_reactor/optimization/inspect_optuna_db.py"
LOG_FILE="${ANALYSIS_DIR}/full_analysis.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   COMPREHENSIVE HPO SWEEP ANALYSIS - LEAVE NO STONE UNTURNED  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Study:${NC} $STUDY_NAME"
echo -e "${GREEN}Storage:${NC} PostgreSQL (DigitalOcean)"
echo -e "${GREEN}Analysis Output:${NC} $ANALYSIS_DIR"
echo ""

# Create output directory
mkdir -p "$ANALYSIS_DIR"

# Start logging
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${YELLOW}[STEP 1/8]${NC} Listing all studies in database..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --list-studies
echo ""

echo -e "${YELLOW}[STEP 2/8]${NC} Analyzing parameter ranges explored..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --analyze-ranges "$STUDY_NAME"
echo ""

echo -e "${YELLOW}[STEP 3/8]${NC} Computing parameter importances (top 20)..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --param-importances \
  --top-k 20 \
  --save-dir "$ANALYSIS_DIR"
echo ""

echo -e "${YELLOW}[STEP 4/8]${NC} Computing parameter directionality (continuous/boolean/categorical)..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --directionality \
  --save-dir "$ANALYSIS_DIR" \
  --min-samples 3
echo ""

echo -e "${YELLOW}[STEP 5/8]${NC} Exporting empirical parameter ranges as YAML..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --as-yaml-param-ranges > "${ANALYSIS_DIR}/empirical_ranges.yaml"
echo -e "${GREEN}✓ Saved to: ${ANALYSIS_DIR}/empirical_ranges.yaml${NC}"
echo ""

echo -e "${YELLOW}[STEP 6/8]${NC} Generating refined search space for next HPO cycle..."
echo "═══════════════════════════════════════════════════════════════"
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --generate-next-search-space \
  --elite-fraction 0.25 \
  --objective-metric val_acc > "${ANALYSIS_DIR}/refined_search_space_v3.yaml"
echo -e "${GREEN}✓ Saved to: ${ANALYSIS_DIR}/refined_search_space_v3.yaml${NC}"
echo ""

echo -e "${YELLOW}[STEP 7/8]${NC} Exporting key parameter interaction data (for contour plots)..."
echo "═══════════════════════════════════════════════════════════════"
# Export key interactions
python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --contour-plot lr batch_size \
  --save-dir "$ANALYSIS_DIR"

python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --contour-plot embed_dim width_mult \
  --save-dir "$ANALYSIS_DIR"

python "$INSPECT_SCRIPT" \
  --storage-url "$STORAGE_URL" \
  --study "$STUDY_NAME" \
  --contour-plot lr weight_decay \
  --save-dir "$ANALYSIS_DIR"
echo ""

echo -e "${YELLOW}[STEP 8/8]${NC} Generating summary report..."
echo "═══════════════════════════════════════════════════════════════"

# Create summary markdown report
cat > "${ANALYSIS_DIR}/ANALYSIS_SUMMARY.md" << 'EOF'
# HPO Sweep Analysis Summary

**Study:** visual_classifier_cnn_vs_context_v2_expanded  
**Analysis Date:** $(date)  
**Database:** PostgreSQL (DigitalOcean)

## Files Generated

### 1. Parameter Importance Analysis
- `param_importances_*.html` - Interactive visualization of parameter impacts
- Console output shows top 20 parameters ranked by effect on validation accuracy

### 2. Directionality Analysis (Effect Direction per Parameter)
- `directionality_continuous_*.csv` - Spearman correlations for numeric params
  - Shows which direction helps (e.g., "increase lr helps" or "increase hurts")
- `directionality_boolean_*.csv` - True vs False comparisons for binary params
- `directionality_categorical_detail_*.csv` - Per-category means
- `directionality_categorical_summary_*.csv` - Best vs worst categories

### 3. Search Space Exports
- `empirical_ranges.yaml` - Observed min/max ranges from completed trials
- `refined_search_space_v3.yaml` - Auto-refined search space for next cycle
  - Expands ranges where elites hit boundaries
  - Narrows ranges where elites cluster
  - Prunes unused categorical choices

### 4. Parameter Interaction Data
- `contour_data_*_lr_vs_batch_size_*.csv` - Learning rate × batch size
- `contour_data_*_embed_dim_vs_width_mult_*.csv` - Embedding dim × width multiplier
- `contour_data_*_lr_vs_weight_decay_*.csv` - Learning rate × regularization

## Key Questions Answered

✅ **Which parameters matter most?** → See parameter importances (Step 3)  
✅ **Which direction to move each parameter?** → See directionality analysis (Step 4)  
✅ **What ranges were actually explored?** → See empirical_ranges.yaml (Step 5)  
✅ **What should the next sweep use?** → See refined_search_space_v3.yaml (Step 6)  
✅ **Are there parameter interactions?** → See contour plot data (Step 7)

## Next Steps

1. Review HTML plots and CSV files in this directory
2. Inspect `refined_search_space_v3.yaml` for next HPO cycle
3. Check directionality CSVs for actionable insights
4. Use contour data to visualize parameter interactions (matplotlib/seaborn)

## Command to Re-run Analysis

\`\`\`bash
bash scripts/analyze_current_hpo_sweep.sh
\`\`\`
EOF

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  ANALYSIS COMPLETE! ✓                          ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Results saved to:${NC} $ANALYSIS_DIR"
echo ""
echo -e "${BLUE}Key files:${NC}"
echo "  📊 param_importances_*.html     - Interactive parameter ranking"
echo "  📈 directionality_*.csv (×4)    - Parameter effect directions"
echo "  🔧 refined_search_space_v3.yaml - Next HPO cycle config"
echo "  📋 empirical_ranges.yaml        - Observed parameter ranges"
echo "  🗺️  contour_data_*.csv (×3)      - Parameter interactions"
echo "  📝 ANALYSIS_SUMMARY.md          - Full summary report"
echo "  📄 full_analysis.log            - Complete analysis log"
echo ""
echo -e "${YELLOW}View the HTML plot:${NC}"
echo "  open ${ANALYSIS_DIR}/param_importances_*.html"
echo ""
echo -e "${YELLOW}Review directionality insights:${NC}"
echo "  cat ${ANALYSIS_DIR}/directionality_continuous_*.csv | column -t -s,"
echo ""
