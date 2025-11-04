# API Reference

## Core Algorithms

### MEMIT
- [`apply_memit_to_model`](algorithms/memit.md#apply_memit_to_model) - Apply MEMIT edits to a model
- [`MEMITHyperParams`](algorithms/memit.md#memithyperparams) - MEMIT hyperparameter configuration

### ROME
- [`apply_rome_to_model`](algorithms/rome.md#apply_rome_to_model) - Apply ROME edits to a model  
- [`ROMEHyperParams`](algorithms/rome.md#romehyperparams) - ROME hyperparameter configuration

### MEND
- [`MendRewriteExecutor`](algorithms/mend.md#mendrewriteexecutor) - MEND model editor
- [`MENDHyperParams`](algorithms/mend.md#mendhyperparams) - MEND hyperparameter configuration

## Utilities

### Generation
- [`generate_fast`](utils/generate.md#generate_fast) - Fast text generation
- [`generate_interactive`](utils/generate.md#generate_interactive) - Interactive generation

### Model Hooks
- [`nethook`](utils/nethook.md) - Neural network activation hooks

### Data Handling
- [`CounterFact`](data/counterfact.md) - CounterFact dataset utilities
- [`zsRE`](data/zsre.md) - Zero-shot relation extraction dataset

## Experiments
- [`evaluate`](experiments/evaluate.md) - Evaluation framework
- [`causal_trace`](experiments/causal_trace.md) - Causal tracing analysis