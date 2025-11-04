"""
Bidirectional Editing Core Module

This module implements sophisticated bidirectional consistency for model editing,
ensuring that edits maintain logical coherence in both forward and reverse directions.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logger = logging.getLogger(__name__)


class BidirectionalConsistencyTracker:
    """
    Tracks and enforces consistency between forward and reverse edits.
    """
    
    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.consistency_threshold = 0.85  # Minimum consistency score required
        self.edit_history = []
        
    def generate_reverse_prompts(self, request: Dict) -> List[Dict]:
        """
        Generate reverse prompts for a given edit request.
        
        Args:
            request: Original edit request with prompt, subject, target_new
            
        Returns:
            List of reverse prompts to ensure bidirectional consistency
        """
        reverse_prompts = []
        
        # Extract components
        subject = request["subject"]
        target_new = request["target_new"]["str"]
        prompt = request["prompt"]
        
        # Generate different types of reverse prompts
        reverse_patterns = [
            "{} is related to",
            "{} corresponds to", 
            "{} is associated with",
            "The counterpart of {} is",
            "{} has a relationship with"
        ]
        
        for pattern in reverse_patterns:
            reverse_request = {
                "prompt": pattern,
                "subject": target_new,
                "target_new": {"str": subject},
                "reverse_of": request,  # Track original request
                "bidirectional": True
            }
            reverse_prompts.append(reverse_request)
            
        return reverse_prompts
    
    def compute_consistency_score(self, 
                                forward_request: Dict, 
                                reverse_request: Dict,
                                edited_model: AutoModelForCausalLM) -> float:
        """
        Compute consistency score between forward and reverse edits.
        
        Args:
            forward_request: Original forward edit request
            reverse_request: Generated reverse edit request  
            edited_model: Model after edits have been applied
            
        Returns:
            Consistency score between 0 and 1
        """
        from ..utils.generate import generate_fast
        
        # Test forward direction
        forward_prompt = forward_request["prompt"].format(forward_request["subject"])
        forward_generation = generate_fast(
            edited_model, self.tokenizer, [forward_prompt], max_out_len=10
        )[0]
        
        # Test reverse direction  
        reverse_prompt = reverse_request["prompt"].format(reverse_request["subject"])
        reverse_generation = generate_fast(
            edited_model, self.tokenizer, [reverse_prompt], max_out_len=10
        )[0]
        
        # Calculate semantic consistency
        forward_contains_target = forward_request["target_new"]["str"].lower() in forward_generation.lower()
        reverse_contains_target = reverse_request["target_new"]["str"].lower() in reverse_generation.lower()
        
        # Basic consistency check - both directions should succeed
        if forward_contains_target and reverse_contains_target:
            consistency_score = 1.0
        elif forward_contains_target or reverse_contains_target:
            consistency_score = 0.5
        else:
            consistency_score = 0.0
            
        return consistency_score
    
    def enforce_consistency_constraint(self, 
                                    delta_weights: Dict[str, torch.Tensor],
                                    consistency_score: float) -> Dict[str, torch.Tensor]:
        """
        Apply consistency constraints to weight updates.
        
        Args:
            delta_weights: Proposed weight changes
            consistency_score: Current consistency score
            
        Returns:
            Adjusted weight changes with consistency enforcement
        """
        if consistency_score < self.consistency_threshold:
            # Apply dampening factor to reduce inconsistent edits
            dampening_factor = consistency_score / self.consistency_threshold
            logger.warning(f"Low consistency score: {consistency_score:.3f}, applying dampening: {dampening_factor:.3f}")
            
            adjusted_weights = {}
            for key, tensor in delta_weights.items():
                adjusted_weights[key] = tensor * dampening_factor
                
            return adjusted_weights
        
        return delta_weights


class BidirectionalEditProcessor:
    """
    Advanced bidirectional edit processor that coordinates between forward and reverse edits.
    """
    
    def __init__(self, 
                 consistency_weight: float = 0.3,
                 max_iterations: int = 3,
                 convergence_threshold: float = 0.01):
        self.consistency_weight = consistency_weight
        self.max_iterations = max_iterations  
        self.convergence_threshold = convergence_threshold
        
    def process_bidirectional_requests(self, 
                                     requests: List[Dict], 
                                     model: AutoModelForCausalLM,
                                     tokenizer: AutoTokenizer) -> Tuple[List[Dict], Dict]:
        """
        Process requests to include bidirectional consistency.
        
        Args:
            requests: Original edit requests
            model: Target model
            tokenizer: Tokenizer
            
        Returns:
            Tuple of (enhanced_requests, consistency_info)
        """
        tracker = BidirectionalConsistencyTracker(model, tokenizer)
        enhanced_requests = []
        consistency_info = {
            "total_requests": len(requests),
            "bidirectional_pairs": 0,
            "consistency_scores": []
        }
        
        # Process each request
        for request in requests:
            enhanced_requests.append(request)
            
            # Generate reverse prompts
            reverse_prompts = tracker.generate_reverse_prompts(request)
            enhanced_requests.extend(reverse_prompts)
            
            consistency_info["bidirectional_pairs"] += len(reverse_prompts)
            
        logger.info(f"Generated {len(enhanced_requests)} total requests from {len(requests)} originals")
        return enhanced_requests, consistency_info
    
    def compute_bidirectional_loss(self, 
                                 forward_loss: torch.Tensor,
                                 reverse_loss: torch.Tensor, 
                                 consistency_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss incorporating bidirectional consistency.
        
        Args:
            forward_loss: Loss from forward edits
            reverse_loss: Loss from reverse edits
            consistency_loss: Loss measuring consistency between directions
            
        Returns:
            Combined bidirectional loss
        """
        total_loss = (
            forward_loss + 
            reverse_loss + 
            self.consistency_weight * consistency_loss
        )
        
        return total_loss


class RelationshipPreservationModule:
    """
    Ensures that complex relationships between entities are preserved during bidirectional editing.
    """
    
    def __init__(self):
        self.relationship_templates = {
            "capital": [
                "{subject} is the capital of {object}",
                "{object} has capital {subject}",
                "The capital city of {object} is {subject}"
            ],
            "profession": [
                "{subject} works as {object}",
                "{object} describes the profession of {subject}",
                "{subject} is employed as {object}"
            ],
            "location": [
                "{subject} is located in {object}",
                "{object} contains {subject}",
                "{subject} can be found in {object}"
            ],
            "sports": [
                "{subject} plays {object}",
                "{object} is played by {subject}",
                "{subject} is a {object} player"
            ]
        }
    
    def detect_relationship_type(self, request: Dict) -> Optional[str]:
        """
        Detect the type of relationship being edited.
        
        Args:
            request: Edit request
            
        Returns:
            Relationship type if detected, None otherwise
        """
        prompt = request["prompt"].lower()
        
        if "capital" in prompt:
            return "capital"
        elif "profession" in prompt or "job" in prompt or "works" in prompt:
            return "profession"  
        elif "located" in prompt or "in" in prompt:
            return "location"
        elif "plays" in prompt or "sport" in prompt:
            return "sports"
            
        return None
    
    def generate_relationship_preserving_edits(self, 
                                             request: Dict, 
                                             relationship_type: str) -> List[Dict]:
        """
        Generate additional edit requests to preserve relationship semantics.
        
        Args:
            request: Original edit request
            relationship_type: Type of relationship detected
            
        Returns:
            List of additional requests to preserve relationship consistency
        """
        if relationship_type not in self.relationship_templates:
            return []
            
        preserving_edits = []
        templates = self.relationship_templates[relationship_type]
        
        subject = request["subject"]
        target = request["target_new"]["str"]
        
        for template in templates:
            # Create variations with subject and object swapped where appropriate
            if "{subject}" in template and "{object}" in template:
                preserving_edit = {
                    "prompt": template.replace("{subject}", "{}").replace("{object}", target),
                    "subject": subject,
                    "target_new": {"str": target},
                    "relationship_type": relationship_type,
                    "preserving_edit": True
                }
                preserving_edits.append(preserving_edit)
                
        return preserving_edits


def create_bidirectional_request_processor() -> BidirectionalEditProcessor:
    """
    Factory function to create a configured bidirectional edit processor.
    
    Returns:
        Configured BidirectionalEditProcessor instance
    """
    return BidirectionalEditProcessor(
        consistency_weight=0.3,
        max_iterations=3,
        convergence_threshold=0.01
    )


def validate_bidirectional_consistency(model: AutoModelForCausalLM,
                                     tokenizer: AutoTokenizer, 
                                     requests: List[Dict],
                                     consistency_threshold: float = 0.8) -> Dict[str, Any]:
    """
    Validate that edits maintain bidirectional consistency.
    
    Args:
        model: Edited model to validate
        tokenizer: Tokenizer
        requests: Original edit requests  
        consistency_threshold: Minimum required consistency score
        
    Returns:
        Validation results with consistency metrics
    """
    tracker = BidirectionalConsistencyTracker(model, tokenizer)
    results = {
        "total_requests": len(requests),
        "consistent_edits": 0,
        "inconsistent_edits": 0,
        "average_consistency": 0.0,
        "consistency_scores": []
    }
    
    for request in requests:
        reverse_prompts = tracker.generate_reverse_prompts(request)
        
        for reverse_request in reverse_prompts:
            consistency_score = tracker.compute_consistency_score(
                request, reverse_request, model
            )
            
            results["consistency_scores"].append(consistency_score)
            
            if consistency_score >= consistency_threshold:
                results["consistent_edits"] += 1
            else:
                results["inconsistent_edits"] += 1
                
    results["average_consistency"] = np.mean(results["consistency_scores"])
    
    logger.info(f"Bidirectional validation complete: {results['consistent_edits']}/{len(results['consistency_scores'])} consistent")
    
    return results