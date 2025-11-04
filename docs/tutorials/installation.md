# Installation Guide

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 16GB+ RAM (for larger models)

## Installation Options

### Option 1: pip install (Recommended)

```bash
pip install bi-memit
```

### Option 2: From Source

```bash
git clone https://github.com/VincentPit/Bi_Memit.git
cd Bi_Memit
pip install -e .
```

### Option 3: Development Installation

```bash
git clone https://github.com/VincentPit/Bi_Memit.git
cd Bi_Memit
pip install -e ".[dev]"
pre-commit install
```

## Conda Environment Setup

For a clean environment, we recommend using conda:

```bash
conda create -n bi-memit python=3.9
conda activate bi-memit
pip install bi-memit
```

## Verify Installation

```python
import src
print(src.__version__)

# Quick test
from src.algorithms import apply_memit_to_model
print("Installation successful!")
```

## GPU Setup

For CUDA support:

```bash
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Troubleshooting

### Common Issues

1. **ImportError**: Make sure all dependencies are installed
2. **CUDA out of memory**: Reduce batch size or use smaller models  
3. **Module not found**: Verify Python path and installation