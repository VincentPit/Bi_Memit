"""
Enhanced ROME with Bidirectional Editing Capabilities

This module extends the original ROME algorithm with sophisticated bidirectional
consistency enforcement for rank-one model editing.
"""

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rome_main import apply_rome_to_model as _original_apply_rome
from .rome_hparams import ROMEHyperParams
from ..bidirectional_core import (
    BidirectionalEditProcessor,
    RelationshipPreservationModule,
    validate_bidirectional_consistency,
    BidirectionalConsistencyTracker
)

import logging
logger = logging.getLogger(__name__)


class BidirectionalROME:
    """
    Enhanced ROME with bidirectional editing capabilities.
    
    This class extends ROME to ensure that rank-one edits maintain
    consistency in both forward and reverse directions.
    """
    
    def __init__(self, 
                 consistency_regularization: float = 0.1,
                 symmetric_updates: bool = True,
                 constraint_enforcement: bool = True):
        """
        Initialize Bidirectional ROME.
        
        Args:
            consistency_regularization: Weight for bidirectional consistency regularization
            symmetric_updates: Whether to enforce symmetric rank-one updates
            constraint_enforcement: Whether to enforce consistency constraints
        """
        self.consistency_regularization = consistency_regularization
        self.symmetric_updates = symmetric_updates
        self.constraint_enforcement = constraint_enforcement
        
        self.processor = BidirectionalEditProcessor(
            consistency_weight=consistency_regularization,
            max_iterations=2,  # ROME typically needs fewer iterations
            convergence_threshold=0.01
        )
        
        self.relationship_module = RelationshipPreservationModule()
    
    def compute_symmetric_rank_one_updates(self,
                                         u_vector: torch.Tensor,
                                         v_vector: torch.Tensor,
                                         reverse_u: torch.Tensor,
                                         reverse_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute symmetric rank-one updates that maintain bidirectional consistency.
        
        Args:
            u_vector: Original u vector for rank-one update
            v_vector: Original v vector for rank-one update  
            reverse_u: u vector for reverse direction
            reverse_v: v vector for reverse direction
            
        Returns:
            Tuple of (symmetrized_u, symmetrized_v)
        """
        if not self.symmetric_updates:
            return u_vector, v_vector
        
        # Compute symmetric updates by averaging forward and reverse components
        # This ensures consistency while maintaining the rank-one structure
        
        # Normalize vectors
        u_norm = torch.norm(u_vector)
        v_norm = torch.norm(v_vector)
        reverse_u_norm = torch.norm(reverse_u)
        reverse_v_norm = torch.norm(reverse_v)
        
        # Compute symmetrized vectors
        alpha = 0.7  # Weight for original direction
        beta = 0.3   # Weight for reverse direction
        
        symmetrized_u = (
            alpha * (u_vector / u_norm) + 
            beta * (reverse_u / reverse_u_norm)
        ) * u_norm
        
        symmetrized_v = (
            alpha * (v_vector / v_norm) +
            beta * (reverse_v / reverse_v_norm) 
        ) * v_norm
        
        return symmetrized_u, symmetrized_v
    
    def enforce_bidirectional_constraints(self,
                                        delta_weights: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                                        consistency_tracker: BidirectionalConsistencyTracker,
                                        model: AutoModelForCausalLM) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Enforce bidirectional consistency constraints on ROME updates.
        
        Args:
            delta_weights: Dictionary of layer names to (u, v) tuples
            consistency_tracker: Bidirectional consistency tracker
            model: Target model
            
        Returns:
            Constrained delta weights
        """
        if not self.constraint_enforcement:
            return delta_weights
        
        constrained_weights = {}
        
        for layer_name, (u_vector, v_vector) in delta_weights.items():
            # Apply consistency constraints - similar to weight dampening but for rank-one updates
            rank_one_update = torch.outer(u_vector, v_vector)
            
            # Check if update maintains reasonable magnitude
            update_norm = torch.norm(rank_one_update)
            max_allowed_norm = 1.0  # Configurable threshold
            
            if update_norm > max_allowed_norm:
                # Scale down the update
                scale_factor = max_allowed_norm / update_norm
                u_vector = u_vector * torch.sqrt(scale_factor)
                v_vector = v_vector * torch.sqrt(scale_factor)
                
                logger.warning(f"Scaled down update for layer {layer_name} by factor {scale_factor:.3f}")
            
            constrained_weights[layer_name] = (u_vector, v_vector)
        
        return constrained_weights
    
    def generate_bidirectional_rome_requests(self,
                                           requests: List[Dict],
                                           model: AutoModelForCausalLM,
                                           tokenizer: AutoTokenizer) -> List[Dict]:
        """
        Generate bidirectional requests optimized for ROME's rank-one structure.
        
        Args:
            requests: Original edit requests
            model: Target model
            tokenizer: Tokenizer
            
        Returns:
            Enhanced requests with bidirectional consistency
        """
        enhanced_requests = []
        
        for request in requests:
            # Add original request
            enhanced_requests.append(request)
            
            # Generate reverse request with ROME-specific optimizations
            subject = request["subject"]
            target_new = request["target_new"]["str"]
            
            # ROME works best with specific, targeted reverse prompts
            reverse_request = {
                "prompt": "{} is associated with",  # More general, works better with ROME
                "subject": target_new,
                "target_new": {"str": subject},
                "bidirectional_pair": True,
                "rome_optimized": True
            }
            enhanced_requests.append(reverse_request)
            
            # Add relationship-preserving requests if applicable
            relationship_type = self.relationship_module.detect_relationship_type(request)
            if relationship_type:
                preserving_requests = self.relationship_module.generate_relationship_preserving_edits(
                    request, relationship_type
                )
                # Limit to 1 additional request for ROME (it's more sensitive to batch size)
                if preserving_requests:
                    enhanced_requests.append(preserving_requests[0])
        
        logger.info(f"Generated {len(enhanced_requests)} ROME-optimized bidirectional requests")
        return enhanced_requests
    
    def apply_regularized_rome_updates(self,
                                     model: AutoModelForCausalLM,
                                     tokenizer: AutoTokenizer,
                                     requests: List[Dict],
                                     hparams: ROMEHyperParams) -> Tuple[AutoModelForCausalLM, Dict]:
        """
        Apply ROME updates with bidirectional regularization.
        
        Args:
            model: Target model
            tokenizer: Tokenizer
            requests: Edit requests
            hparams: ROME hyperparameters
            
        Returns:
            Tuple of (edited_model, update_info)
        """
        # Generate bidirectional requests
        bidirectional_requests = self.generate_bidirectional_rome_requests(
            requests, model, tokenizer
        )
        
        # Separate forward and reverse requests
        forward_requests = [r for r in bidirectional_requests if not r.get("bidirectional_pair", False)]
        reverse_requests = [r for r in bidirectional_requests if r.get("bidirectional_pair", False)]
        
        # Apply ROME to forward requests
        forward_model, forward_weights = _original_apply_rome(
            model=deepcopy(model),
            tok=tokenizer,
            requests=forward_requests,
            hparams=hparams,
            copy=False,
            return_orig_weights=True
        )
        
        # Apply ROME to reverse requests on original model
        if reverse_requests:
            reverse_model, reverse_weights = _original_apply_rome(
                model=deepcopy(model),
                tok=tokenizer,
                requests=reverse_requests,
                hparams=hparams,
                copy=False,
                return_orig_weights=True
            )
        else:
            reverse_model = model
            reverse_weights = {}
        
        # Combine updates with bidirectional consistency
        consistency_tracker = BidirectionalConsistencyTracker(model, tokenizer)
        
        # For ROME, we need to handle the rank-one structure carefully
        final_model = deepcopy(model)
        
        # Apply forward updates
        with torch.no_grad():
            for param_name, param in final_model.named_parameters():
                if param_name in forward_weights:
                    param.add_(forward_weights[param_name])
        
        # Apply regularized reverse updates
        if reverse_weights:
            with torch.no_grad():
                regularization_strength = self.consistency_regularization
                for param_name, param in final_model.named_parameters():
                    if param_name in reverse_weights:
                        # Apply reverse update with regularization
                        param.add_(reverse_weights[param_name] * regularization_strength)
        
        update_info = {
            "forward_requests": len(forward_requests),
            "reverse_requests": len(reverse_requests),
            "total_requests": len(bidirectional_requests),
            "regularization_applied": len(reverse_weights) > 0
        }
        
        return final_model, update_info


def apply_bidirectional_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    cache_template: Optional[str] = None,
    bidirectional_config: Optional[Dict] = None,
) -> Tuple[AutoModelForCausalLM, Dict]:
    """
    Apply ROME with enhanced bidirectional consistency.
    
    Args:
        model: The original causal language model to apply changes to
        tok: The tokenizer used with the model
        requests: List of dictionaries containing the update requests
        hparams: Hyperparameters used for ROME updates
        copy: If True, creates a deep copy of the model to apply changes
        return_orig_weights: If True, returns the original weights that were modified
        cache_template: Optional cache template for intermediate results
        bidirectional_config: Configuration for bidirectional enhancements
        
    Returns:
        Tuple of (updated_model, info_dict)
    """
    # Default bidirectional configuration for ROME
    if bidirectional_config is None:
        bidirectional_config = {
            "consistency_regularization": 0.1,  # Lower for ROME's precision
            "symmetric_updates": True,
            "constraint_enforcement": True,
            "validation_threshold": 0.85
        }
    
    logger.info(f"Applying Bidirectional ROME to model with {len(requests)} requests")
    
    # Initialize bidirectional ROME
    bi_rome = BidirectionalROME(
        consistency_regularization=bidirectional_config["consistency_regularization"],
        symmetric_updates=bidirectional_config["symmetric_updates"],
        constraint_enforcement=bidirectional_config["constraint_enforcement"]
    )
    
    # Create model copy if requested
    if copy:
        model = deepcopy(model)
    
    # Apply bidirectional ROME
    edited_model, update_info = bi_rome.apply_regularized_rome_updates(
        model=model,
        tokenizer=tok,
        requests=requests,
        hparams=hparams
    )
    
    # Validate bidirectional consistency
    validation_results = validate_bidirectional_consistency(
        edited_model, tok, requests,
        consistency_threshold=bidirectional_config["validation_threshold"]
    )
    
    # Prepare return information
    return_info = {
        "bidirectional_config": bidirectional_config,
        "update_info": update_info,
        "validation_results": validation_results,
        "consistency_score": validation_results["average_consistency"],
        "requests_processed": len(requests)
    }
    
    if return_orig_weights:
        # Get original weights if requested
        _, orig_weights = _original_apply_rome(
            model=model,
            tok=tok,
            requests=[],  # Empty to just get original weights
            hparams=hparams,
            copy=False,
            return_orig_weights=True
        )
        return_info["original_weights"] = orig_weights
    
    logger.info(f"Bidirectional ROME completed. Consistency: {validation_results['average_consistency']:.4f}")
    
    return edited_model, return_info


def compute_rome_bidirectional_metrics(model: AutoModelForCausalLM,
                                     tokenizer: AutoTokenizer, 
                                     original_requests: List[Dict],
                                     edited_model: AutoModelForCausalLM) -> Dict:
    """
    Compute comprehensive metrics for bidirectional ROME edits.
    
    Args:
        model: Original model
        tokenizer: Tokenizer
        original_requests: Original edit requests
        edited_model: Model after bidirectional ROME edits
        
    Returns:
        Dictionary of bidirectional metrics
    """
    tracker = BidirectionalConsistencyTracker(model, tokenizer)
    
    metrics = {
        "precision_maintained": 0,
        "consistency_scores": [],
        "relationship_preservation": 0,
        "edit_success_rate": 0
    }
    
    from ..utils.generate import generate_fast
    
    successful_edits = 0
    
    for request in original_requests:
        # Test forward edit success
        prompt = request["prompt"].format(request["subject"])
        generation = generate_fast(edited_model, tokenizer, [prompt], max_out_len=10)[0]
        
        if request["target_new"]["str"].lower() in generation.lower():
            successful_edits += 1
        
        # Test bidirectional consistency
        reverse_requests = tracker.generate_reverse_prompts(request)
        for reverse_request in reverse_requests:
            consistency_score = tracker.compute_consistency_score(
                request, reverse_request, edited_model
            )
            metrics["consistency_scores"].append(consistency_score)
    
    metrics["edit_success_rate"] = successful_edits / len(original_requests)
    metrics["average_consistency"] = np.mean(metrics["consistency_scores"]) if metrics["consistency_scores"] else 0.0
    
    return metrics