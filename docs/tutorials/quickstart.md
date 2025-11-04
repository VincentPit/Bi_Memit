# Quick Start Guide

This guide will help you get started with Bi-MEMIT in just a few minutes.

## Basic Usage

### 1. Load a Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.algorithms import apply_memit_to_model, MEMITHyperParams

# Load model and tokenizer
model_name = "gpt2-medium"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
```

### 2. Define Edit Requests

```python
# Define the facts you want to edit
requests = [
    {
        "prompt": "{} plays the sport of",
        "subject": "LeBron James", 
        "target_new": {"str": "football"}
    },
    {
        "prompt": "{} is the capital of",
        "subject": "France",
        "target_new": {"str": "Lyon"}
    }
]
```

### 3. Load Hyperparameters

```python
# Load MEMIT hyperparameters
hparams = MEMITHyperParams.from_hparams("config/hparams/MEMIT/gpt2-medium.json")
```

### 4. Apply Edits

```python
# Apply MEMIT to edit the model
edited_model, _ = apply_memit_to_model(
    model=model,
    tok=tokenizer,
    requests=requests, 
    hparams=hparams,
    copy=True  # Keep original model intact
)
```

### 5. Test the Edits

```python
from src.utils.generate import generate_fast

# Test the edited model
prompt = "LeBron James plays the sport of"
generated = generate_fast(edited_model, tokenizer, [prompt], max_out_len=10)
print(f"Edited: {generated[0]}")

# Compare with original
original_gen = generate_fast(model, tokenizer, [prompt], max_out_len=10)  
print(f"Original: {original_gen[0]}")
```

## Algorithm Comparison

### MEMIT vs ROME vs MEND

| Algorithm | Best For | Batch Size | Memory Usage |
|-----------|----------|------------|--------------|
| MEMIT     | Mass edits (1000s) | Large | High |
| ROME      | Single precise edits | 1 | Low |
| MEND      | Learned editing | Medium | Medium |

### When to Use Each Algorithm

- **MEMIT**: When you need to edit thousands of facts efficiently
- **ROME**: For single, high-precision edits with minimal side effects
- **MEND**: When you have training data and want to learn editing patterns

## Next Steps

- [Detailed Examples](../examples/index.md)
- [API Reference](../api/index.md)
- [Advanced Configuration](advanced/configuration.md)