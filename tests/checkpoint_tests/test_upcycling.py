import os
import shutil
import tempfile
import unittest
import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM, AutoConfig, AutoModelForCausalLM

# Add workspace to path
import sys
sys.path.append("/home/conan/workspace/llm_training")

from models.spectra_model import SPECTRAForCausalLM, SPECTRAConfig, SPECTRAMoE

class TestUpcycling(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_llama_upcycling(self):
        print("\n=== Testing Llama Upcycling ===")
        # 1. Create and save a dummy Llama model
        config = LlamaConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128
        )
        model = LlamaForCausalLM(config)
        save_path = os.path.join(self.tmp_dir, "dummy_llama")
        model.save_pretrained(save_path)
        print(f"Saved dummy Llama model to {save_path}")

        # 2. Define SPECTRA config for upcycling
        spectra_config = SPECTRAConfig()
        spectra_config.text_config.n_routed_experts = 4
        spectra_config.text_config.num_experts_per_tok = 2
        # Ensure compatible dimensions
        spectra_config.text_config.hidden_size = config.hidden_size
        spectra_config.text_config.intermediate_size = config.intermediate_size
        spectra_config.text_config.num_hidden_layers = config.num_hidden_layers

        # 3. Load with SPECTRAForCausalLM + force_upcycle=True
        print("Loading with SPECTRAForCausalLM...")
        spectra_model = SPECTRAForCausalLM.from_pretrained(
            save_path,
            config=spectra_config,
            force_upcycle=True,
            local_files_only=True
        )

        # 4. Verify Upcycling
        print("Verifying layers...")
        if hasattr(spectra_model, 'language_model'):
             layers = spectra_model.language_model.layers
        elif hasattr(spectra_model, 'model'):
             layers = spectra_model.model.layers
        else:
             layers = spectra_model.layers
        self.assertTrue(len(layers) > 0)
        
        # Check first layer
        first_layer = layers[0]
        self.assertTrue(hasattr(first_layer, 'mlp'), "First layer should have 'mlp' attribute")
        
        # Check if MLP was converted to SPECTRAMoE
        self.assertIsInstance(first_layer.mlp, SPECTRAMoE, "MLP should be converted to SPECTRAMoE")
        print("✅ Llama upcycling successful: MLP is now SPECTRAMoE")

    def test_gemma_structure_simulation(self):
        print("\n=== Testing Gemma Structure Simulation ===")
        # Gemma uses 'self_attn' and 'mlp' inside layers, similar to Llama.
        # So essentially testing Llama *is* testing the structure Gemma uses regarding MLP naming.
        # But let's verify if 'feed_forward' vs 'mlp' handling is robust.
        
        # We can simulate a model with 'feed_forward' instead of 'mlp'
        # But standard HF Llama/Gemma use 'mlp'. T5 uses 'DenseReluDense'. 
        
        pass

if __name__ == "__main__":
    unittest.main()
