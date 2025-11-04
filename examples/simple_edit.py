#!/usr/bin/env python3
"""
Simple example of using Bi-MEMIT for model editing

This script demonstrates how to edit a GPT-2 model to change factual knowledge.
"""

import sys
import json
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.algorithms.memit import apply_memit_to_model, MEMITHyperParams
    from src.utils.generate import generate_fast
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install dependencies: pip install -r requirements.txt")
    sys.exit(1)


def main():
    print("🧠 Bi-MEMIT Example: Editing Model Knowledge")
    print("=" * 50)
    
    # Load model and tokenizer
    print("📦 Loading GPT-2 model...")
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Define edit requests
    print("✏️  Defining edit requests...")
    requests = [
        {
            "prompt": "{} is the capital of",
            "subject": "France", 
            "target_new": {"str": "Lyon"}
        },
        {
            "prompt": "{} plays the sport of",
            "subject": "LeBron James",
            "target_new": {"str": "tennis"}  
        }
    ]
    
    print(f"   - Editing {len(requests)} facts")
    for req in requests:
        print(f"   - '{req['subject']}' → '{req['target_new']['str']}'")
    
    # Test original model
    print("\n🔍 Testing original model...")
    for req in requests:
        prompt = req["prompt"].format(req["subject"])
        original_output = generate_fast(model, tokenizer, [prompt], max_out_len=5)
        print(f"   Original: {prompt} → {original_output[0].split(prompt)[1].strip()}")
    
    # Load hyperparameters
    print("\n⚙️  Loading hyperparameters...")
    hparams = MEMITHyperParams(
        layers=[5, 6, 7, 8, 9, 10],
        layer_selection="all", 
        fact_token="subject_last",
        v_num_grad_steps=20,
        v_lr=1e-1,
        v_loss_layer=31,
        kv_cache_dtype="float32"
    )
    
    # Apply MEMIT edits
    print("🔧 Applying MEMIT edits...")
    try:
        edited_model, orig_weights = apply_memit_to_model(
            model=model,
            tok=tokenizer, 
            requests=requests,
            hparams=hparams,
            copy=True,
            return_orig_weights=True
        )
        print("   ✅ Edits applied successfully!")
    except Exception as e:
        print(f"   ❌ Error applying edits: {e}")
        return
    
    # Test edited model
    print("\n🎯 Testing edited model...")
    for req in requests:
        prompt = req["prompt"].format(req["subject"])
        edited_output = generate_fast(edited_model, tokenizer, [prompt], max_out_len=5)
        print(f"   Edited:   {prompt} → {edited_output[0].split(prompt)[1].strip()}")
    
    print("\n🎉 Example completed successfully!")
    print("\nTry running with different edit requests or models!")


if __name__ == "__main__":
    main()