"""
Command Line Interface for Bi-MEMIT

This module provides a CLI for running model editing experiments.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

from src.algorithms import apply_memit_to_model, MEMITHyperParams
from src.experiments.evaluate import evaluate_model
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_requests(file_path: str) -> List[Dict]:
    """Load edit requests from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)


def save_results(results: Dict, output_path: str):
    """Save results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def cmd_edit(args):
    """Run model editing with specified algorithm"""
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    logger.info(f"Loading requests from: {args.requests}")
    requests = load_requests(args.requests)
    
    logger.info(f"Loading hyperparameters from: {args.hparams}")
    hparams = MEMITHyperParams.from_hparams(args.hparams)
    
    logger.info("Applying edits to model...")
    edited_model, orig_weights = apply_memit_to_model(
        model=model,
        tok=tokenizer,
        requests=requests,
        hparams=hparams,
        copy=True,
        return_orig_weights=True
    )
    
    # Save edited model if specified
    if args.output_model:
        logger.info(f"Saving edited model to: {args.output_model}")
        edited_model.save_pretrained(args.output_model)
        tokenizer.save_pretrained(args.output_model)
    
    logger.info("Model editing completed successfully!")


def cmd_evaluate(args):
    """Run evaluation on edited model"""
    logger.info("Running model evaluation...")
    results = evaluate_model(
        model_name=args.model_name,
        algorithm=args.algorithm,
        dataset=args.dataset,
        num_edits=args.num_edits,
        hparams_path=args.hparams
    )
    
    if args.output:
        save_results(results, args.output)
        logger.info(f"Results saved to: {args.output}")
    else:
        print(json.dumps(results, indent=2))


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Bi-MEMIT: Model Editing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Edit command
    edit_parser = subparsers.add_parser('edit', help='Edit a model')
    edit_parser.add_argument('--model-name', required=True, 
                           help='Model name or path')
    edit_parser.add_argument('--algorithm', default='MEMIT',
                           choices=['MEMIT', 'ROME', 'MEND'],
                           help='Editing algorithm to use')
    edit_parser.add_argument('--requests', required=True,
                           help='JSON file with edit requests')
    edit_parser.add_argument('--hparams', required=True,
                           help='Hyperparameters file')
    edit_parser.add_argument('--output-model',
                           help='Path to save edited model')
    edit_parser.set_defaults(func=cmd_edit)
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--model-name', required=True,
                           help='Model name or path')
    eval_parser.add_argument('--algorithm', default='MEMIT',
                           choices=['MEMIT', 'ROME', 'MEND'],
                           help='Editing algorithm used')
    eval_parser.add_argument('--dataset', default='counterfact',
                           choices=['counterfact', 'zsre'],
                           help='Evaluation dataset')
    eval_parser.add_argument('--num-edits', type=int, default=100,
                           help='Number of edits to evaluate')
    eval_parser.add_argument('--hparams', required=True,
                           help='Hyperparameters file')
    eval_parser.add_argument('--output',
                           help='Output file for results')
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())