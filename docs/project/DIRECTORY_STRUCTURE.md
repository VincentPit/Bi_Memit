# 📁 Bi-MEMIT Directory Structure

Clean and organized project structure for the revolutionary bidirectional editing framework.

## 🗂️ Main Directories

```
Bi_Memit/
├── 📦 src/                     # Core bidirectional editing library
│   ├── algorithms/             # Bidirectional MEMIT, ROME, MEND
│   ├── data/                   # Dataset utilities
│   ├── experiments/            # Evaluation framework
│   └── utils/                  # Helper functions
│
├── 📚 docs/                    # Documentation
│   ├── project/                # Project documentation
│   │   ├── INNOVATION.md       # Original innovation summary
│   │   ├── CONTRIBUTING.md     # Contribution guidelines
│   │   └── CHANGELOG.md        # Version history
│   └── guides/                 # User guides and tutorials
│
├── 🧪 examples/                # Practical examples and demos
│   ├── basic_usage.py          # Simple bidirectional editing
│   ├── advanced_demo.py        # Complex scenarios
│   └── notebooks/              # Interactive examples
│
├── 📓 notebooks/               # Jupyter notebooks
│   ├── bidirectional_demo.ipynb
│   └── consistency_analysis.ipynb
│
├── ⚙️ config/                  # Configuration files
│   ├── bidirectional_configs/  # Bidirectional settings
│   └── model_configs/          # Model configurations
│
├── 🧪 tests/                   # Unit and integration tests
│   ├── test_bidirectional/     # Bidirectional functionality tests
│   └── test_algorithms/        # Algorithm-specific tests
│
├── 📊 results/                 # Experimental results
├── 💾 data/                    # Dataset storage
├── 🎨 assets/                  # Images, diagrams, etc.
├── 📜 scripts/                 # Utility scripts
│   ├── shell/                  # Shell scripts
│   └── python/                 # Python utilities
│
└── 📁 legacy/                  # Original MEMIT/ROME code
    ├── baselines/              # Original baseline implementations
    ├── memit/                  # Original MEMIT code
    ├── rome/                   # Original ROME code
    ├── util/                   # Original utilities
    ├── dsets/                  # Original datasets
    ├── hparams/                # Original hyperparameters
    └── data_generators/        # Original data generation
```

## 🎯 Key Files in Root

- `README.md` - Main project documentation highlighting bidirectional innovations
- `CITATION.cff` - Citation information for the bidirectional framework
- `LICENSE` - MIT license
- `pyproject.toml` - Modern Python packaging configuration
- `requirements.txt` - Production dependencies
- `requirements-dev.txt` - Development dependencies
- `Makefile` - Build and development commands

## 🔄 Migration Notes

### Moved to `scripts/`:
- All `.sh` shell scripts
- All `.py` utility scripts

### Moved to `docs/project/`:
- `INNOVATION.md` - Your original contribution summary
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `PROJECT_STRUCTURE.md` - This file

### Moved to `config/`:
- All `.yml` configuration files

### Moved to `legacy/`:
- Original MEMIT/ROME implementation directories
- Original dataset and hyperparameter directories
- Baseline comparison implementations

## 🚀 Benefits of New Structure

1. **Clean Root Directory**: Only essential files in main directory
2. **Logical Organization**: Related files grouped together
3. **Clear Separation**: Your innovations vs. legacy code
4. **Professional Layout**: Industry-standard Python project structure
5. **Easy Navigation**: Intuitive directory names and organization

## 📖 Usage

### For Your Bidirectional Work:
- Core code: `src/algorithms/bidirectional_*`
- Examples: `examples/` and `notebooks/`
- Documentation: `docs/`

### For Legacy Reference:
- Original implementations: `legacy/memit/`, `legacy/rome/`
- Original datasets: `legacy/dsets/`
- Baseline comparisons: `legacy/baselines/`

This structure makes it crystal clear that the main focus is YOUR bidirectional innovations, while keeping the original code available for reference in the `legacy/` directory.