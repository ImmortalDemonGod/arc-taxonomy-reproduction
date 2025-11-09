# Neural Affinity Framework for ARC - Reproduction Package

**Paper:** Ingram & Merrit (2025) - *An Empirical Answer to re-arc*  
**Competition:** ARC Prize 2025 Paper Award Submission

## 📄 Paper

[**Ingram_Merrit_2025_Neural_Affinity_Framework_ARC.pdf**](./Ingram_Merrit_2025_Neural_Affinity_Framework_ARC.pdf) - Full 58-page paper

## 🎯 Key Contributions

1. **First validated 9-category taxonomy** (97.5% accuracy on re-arc)
2. **Compositional Gap quantification** (69.5% of tasks show cell vs grid dissociation)
3. **External validation** on ViTARC (400 independent models, p<0.001)
4. **Neural Affinity Framework** - diagnostic tool for architectural suitability

## 📁 Structure

- `src/` - 1.7M parameter Transformer model
- `scripts/` - Taxonomy generation, analysis, figure creation
- `data/` - Taxonomy classifications and external validation data
- `docs/` - Comprehensive documentation
- `figures/` - All paper figures (reproducible)
- `weights/` - Pre-trained model checkpoints
- `outputs/` - Experiment results

## 🚀 Quick Start

See [**QUICKSTART.md**](./QUICKSTART.md) for reproduction instructions.

## 📊 Results

- **re-arc taxonomy:** 97.5% (390/400 tasks)
- **ARC-AGI-2:** 0.34% (diagnostic model, not solver)
- **External validation:** Affinity predictions confirmed (Cohen's d=0.726)

## 📝 Citation

```bibtex
@article{ingram2025neuralaffinity,
  title={An Empirical Answer to re-arc: A Systematic Taxonomy, 
         Curriculum Analysis, and Neural Affinity Framework},
  author={Ingram, Miguel and Merrit, Arthur},
  year={2025}
}
```

## 📧 Contact

For questions: Open an issue on GitHub

---
**Last Updated:** November 9, 2025
