"""
Test suite for Bi-MEMIT algorithms
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.algorithms import apply_memit_to_model, MEMITHyperParams


class TestMemit:
    """Test cases for MEMIT algorithm"""

    @pytest.fixture
    def model_and_tokenizer(self):
        """Load a small model for testing"""
        model_name = "gpt2"
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    @pytest.fixture
    def sample_requests(self):
        """Sample edit requests for testing"""
        return [
            {
                "prompt": "{} plays the sport of",
                "subject": "LeBron James",
                "target_new": {"str": "tennis"}
            }
        ]

    @pytest.fixture
    def hparams(self):
        """Sample hyperparameters for testing"""
        return MEMITHyperParams(
            layers=[5, 6, 7, 8, 9, 10],
            layer_selection="all",
            fact_token="subject_last",
            v_num_grad_steps=5,
            v_lr=1e-1,
            v_loss_layer=10,
            kv_cache_dtype="float32"
        )

    def test_memit_basic_functionality(self, model_and_tokenizer, sample_requests, hparams):
        """Test basic MEMIT functionality"""
        model, tokenizer = model_and_tokenizer
        
        # Apply MEMIT
        edited_model, orig_weights = apply_memit_to_model(
            model=model,
            tok=tokenizer,
            requests=sample_requests,
            hparams=hparams,
            copy=True,
            return_orig_weights=True
        )
        
        # Check that model was modified
        assert edited_model is not None
        assert orig_weights is not None
        
        # Ensure original model parameters are preserved
        assert not torch.equal(
            list(model.parameters())[0],
            list(edited_model.parameters())[0]
        )

    def test_memit_preserves_model_structure(self, model_and_tokenizer, sample_requests, hparams):
        """Test that MEMIT preserves model architecture"""
        model, tokenizer = model_and_tokenizer
        original_config = model.config
        
        edited_model, _ = apply_memit_to_model(
            model=model,
            tok=tokenizer,
            requests=sample_requests,
            hparams=hparams,
            copy=True
        )
        
        # Check that model structure is preserved
        assert edited_model.config.to_dict() == original_config.to_dict()
        assert len(list(edited_model.parameters())) == len(list(model.parameters()))