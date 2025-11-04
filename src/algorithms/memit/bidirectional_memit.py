"""
Enhanced MEMIT with Bidirectional Editing Capabilities

This module extends the original MEMIT algorithm with sophisticated bidirectional
consistency enforcement and relationship preservation.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

from .memit_main import apply_memit_to_model as _original_apply_memit
from .memit_hparams import MEMITHyperParams
from .bidirectional_core import (
    BidirectionalEditProcessor,
    RelationshipPreservationModule, 
    validate_bidirectional_consistency
)

import logging
logger = logging.getLogger(__name__)


class BidirectionalMEMIT:
    """
    Enhanced MEMIT with bidirectional editing capabilities.
    """
    
    def __init__(self, 
                 consistency_weight: float = 0.25,
                 relationship_preservation: bool = True,
                 adaptive_threshold: bool = True):
        """
        Initialize Bidirectional MEMIT.
        
        Args:
            consistency_weight: Weight for bidirectional consistency loss
            relationship_preservation: Whether to preserve relationship semantics
            adaptive_threshold: Whether to use adaptive consistency thresholds
        """
        self.consistency_weight = consistency_weight
        self.relationship_preservation = relationship_preservation
        self.adaptive_threshold = adaptive_threshold
        
        self.processor = BidirectionalEditProcessor(
            consistency_weight=consistency_weight,
            max_iterations=5,
            convergence_threshold=0.005
        )
        
        if relationship_preservation:
            self.relationship_module = RelationshipPreservationModule()
        
    def enhance_requests_with_bidirectional_consistency(self, 
                                                       requests: List[Dict],
                                                       model: AutoModelForCausalLM,
                                                       tokenizer: AutoTokenizer) -> Tuple[List[Dict], Dict]:
        """
        Enhance edit requests with bidirectional consistency requirements.
        
        Args:
            requests: Original edit requests
            model: Target model
            tokenizer: Tokenizer
            
        Returns:
            Tuple of (enhanced_requests, metadata)
        """
        logger.info(f"Enhancing {len(requests)} requests with bidirectional consistency")
        
        enhanced_requests = []
        metadata = {
            "original_count": len(requests),
            "bidirectional_pairs": 0,
            "relationship_preserving": 0,
            "total_enhanced": 0
        }
        
        for request in requests:
            # Add original request
            enhanced_requests.append(request)
            
            # Process with bidirectional processor
            bidirectional_requests, _ = self.processor.process_bidirectional_requests(
                [request], model, tokenizer
            )
            
            # Add bidirectional requests (excluding the original)
            new_requests = bidirectional_requests[1:]  # Skip first (original)
            enhanced_requests.extend(new_requests)
            metadata["bidirectional_pairs"] += len(new_requests)
            
            # Add relationship preservation if enabled
            if self.relationship_preservation:
                relationship_type = self.relationship_module.detect_relationship_type(request)
                if relationship_type:
                    preserving_requests = self.relationship_module.generate_relationship_preserving_edits(
                        request, relationship_type
                    )
                    enhanced_requests.extend(preserving_requests)
                    metadata["relationship_preserving"] += len(preserving_requests)
        
        metadata["total_enhanced"] = len(enhanced_requests)
        
        logger.info(f"Enhanced to {len(enhanced_requests)} total requests "
                   f"({metadata['bidirectional_pairs']} bidirectional, "
                   f"{metadata['relationship_preserving']} relationship-preserving)")
        
        return enhanced_requests, metadata
    
    def compute_bidirectional_loss(self, 
                                 forward_outputs: torch.Tensor,
                                 reverse_outputs: torch.Tensor,
                                 forward_targets: torch.Tensor,
                                 reverse_targets: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss incorporating bidirectional consistency.
        
        Args:
            forward_outputs: Model outputs for forward edits
            reverse_outputs: Model outputs for reverse edits  
            forward_targets: Target tokens for forward edits
            reverse_targets: Target tokens for reverse edits
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Standard cross-entropy losses
        forward_loss = torch.nn.functional.cross_entropy(forward_outputs, forward_targets)
        reverse_loss = torch.nn.functional.cross_entropy(reverse_outputs, reverse_targets)
        
        # Consistency loss - measure similarity between forward and reverse representations
        consistency_loss = self._compute_consistency_loss(forward_outputs, reverse_outputs)
        
        # Combine losses
        total_loss = forward_loss + reverse_loss + self.consistency_weight * consistency_loss
        
        loss_components = {
            "forward_loss": forward_loss.item(),
            "reverse_loss": reverse_loss.item(), 
            "consistency_loss": consistency_loss.item(),
            "total_loss": total_loss.item()
        }
        
        return total_loss, loss_components
    
    def _compute_consistency_loss(self, 
                                forward_outputs: torch.Tensor, 
                                reverse_outputs: torch.Tensor) -> torch.Tensor:
        """
        Compute consistency loss between forward and reverse outputs.
        
        Args:
            forward_outputs: Forward direction outputs
            reverse_outputs: Reverse direction outputs
            
        Returns:
            Consistency loss tensor
        """
        # Normalize outputs to probability distributions
        forward_probs = torch.nn.functional.softmax(forward_outputs, dim=-1)
        reverse_probs = torch.nn.functional.softmax(reverse_outputs, dim=-1)
        
        # KL divergence as consistency measure
        consistency_loss = torch.nn.functional.kl_div(
            torch.log(forward_probs), reverse_probs, reduction='batchmean'
        )
        
        return consistency_loss
    
    def iterative_bidirectional_refinement(self,
                                         model: AutoModelForCausalLM,
                                         tokenizer: AutoTokenizer,
                                         requests: List[Dict],
                                         hparams: MEMITHyperParams,
                                         max_iterations: int = 3) -> Tuple[AutoModelForCausalLM, Dict]:
        """
        Perform iterative refinement to improve bidirectional consistency.
        
        Args:
            model: Model to edit
            tokenizer: Tokenizer
            requests: Edit requests
            hparams: MEMIT hyperparameters
            max_iterations: Maximum refinement iterations
            
        Returns:
            Tuple of (refined_model, refinement_info)
        """
        current_model = model
        refinement_info = {
            "iterations": 0,
            "consistency_scores": [],
            "convergence_achieved": False
        }
        
        for iteration in range(max_iterations):
            logger.info(f"Bidirectional refinement iteration {iteration + 1}/{max_iterations}")
            
            # Enhance requests for current iteration
            enhanced_requests, _ = self.enhance_requests_with_bidirectional_consistency(
                requests, current_model, tokenizer
            )
            
            # Apply MEMIT with enhanced requests
            refined_model, _ = _original_apply_memit(
                model=current_model,
                tok=tokenizer,
                requests=enhanced_requests,
                hparams=hparams,
                copy=True
            )
            
            # Validate consistency
            validation_results = validate_bidirectional_consistency(
                refined_model, tokenizer, requests
            )
            
            consistency_score = validation_results["average_consistency"]
            refinement_info["consistency_scores"].append(consistency_score)
            refinement_info["iterations"] = iteration + 1
            
            logger.info(f"Iteration {iteration + 1} consistency score: {consistency_score:.4f}")
            
            # Check for convergence
            if iteration > 0:
                improvement = consistency_score - refinement_info["consistency_scores"][-2]
                if improvement < 0.01:  # Convergence threshold
                    logger.info("Convergence achieved")
                    refinement_info["convergence_achieved"] = True
                    break
            
            current_model = refined_model
        
        return current_model, refinement_info


def apply_bidirectional_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    bidirectional_config: Optional[Dict] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Apply MEMIT with enhanced bidirectional consistency.
    
    Args:
        model: The original causal language model to apply changes to
        tok: The tokenizer used with the model
        requests: List of dictionaries containing the update requests
        hparams: Hyperparameters used for MEMIT updates
        copy: If True, creates a deep copy of the model to apply changes
        return_orig_weights: If True, returns the original weights that were modified
        cache_template: Optional cache template for intermediate results
        bidirectional_config: Configuration for bidirectional enhancements
        
    Returns:
        Tuple of (updated_model, info_dict)
    """
    # Default bidirectional configuration
    if bidirectional_config is None:
        bidirectional_config = {
            "consistency_weight": 0.25,
            "relationship_preservation": True,
            "adaptive_threshold": True,
            "iterative_refinement": True,
            "max_iterations": 3
        }
    
    logger.info(f"Applying Bidirectional MEMIT to model with {len(requests)} requests")
    
    # Initialize bidirectional MEMIT
    bi_memit = BidirectionalMEMIT(
        consistency_weight=bidirectional_config["consistency_weight"],
        relationship_preservation=bidirectional_config["relationship_preservation"],
        adaptive_threshold=bidirectional_config["adaptive_threshold"]
    )
    
    # Create model copy if requested
    if copy:
        model = deepcopy(model)
    
    # Apply iterative refinement if enabled
    if bidirectional_config.get("iterative_refinement", True):
        edited_model, refinement_info = bi_memit.iterative_bidirectional_refinement(
            model=model,
            tokenizer=tok,
            requests=requests,
            hparams=hparams,
            max_iterations=bidirectional_config.get("max_iterations", 3)
        )
    else:
        # Single-pass bidirectional enhancement
        enhanced_requests, enhancement_info = bi_memit.enhance_requests_with_bidirectional_consistency(
            requests, model, tok
        )
        
        edited_model, orig_weights = _original_apply_memit(
            model=model,
            tok=tok,
            requests=enhanced_requests,
            hparams=hparams,
            copy=False,
            return_orig_weights=return_orig_weights,
            cache_template=cache_template
        )
        
        refinement_info = {
            "enhancement_info": enhancement_info,
            "single_pass": True
        }
    
    # Final validation
    final_validation = validate_bidirectional_consistency(
        edited_model, tok, requests
    )
    
    # Prepare return information
    return_info = {
        "bidirectional_config": bidirectional_config,
        "refinement_info": refinement_info,
        "final_validation": final_validation,
        "requests_processed": len(requests)
    }
    
    if return_orig_weights:
        # Get original weights if requested
        _, orig_weights = _original_apply_memit(
            model=model,
            tok=tok,
            requests=[],  # Empty to just get original weights
            hparams=hparams,
            copy=False,
            return_orig_weights=True
        )
        return_info["original_weights"] = orig_weights
    
    logger.info(f"Bidirectional MEMIT completed. Final consistency: {final_validation['average_consistency']:.4f}")
    
    return edited_model, return_info