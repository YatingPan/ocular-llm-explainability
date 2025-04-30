"""
Custom LRP implementation for RETFound regression model.
Modified to handle regression tasks instead of classification.
"""

import numpy as np
import torch

class CustomLRP:
    """
    LRP implementation for regression models
    """
    
    def __init__(self, model):
        self.model = model
        self.model.train()  # Keep in train mode for gradient tracking
        
    def generate_LRP(self, input, method="transformer_attribution", is_ablation=False, start_layer=0):
        """
        Generate LRP attribution for regression output.
        
        Args:
            input: Input tensor [B,C,H,W]
            method: Attribution method ("transformer_attribution", "rollout", etc.)
            is_ablation: Whether to use ablation
            start_layer: First layer for rollout
            
        Returns:
            Attribution map for the patches
        """
        # Make sure input requires gradients
        input.requires_grad_(True)
        
        # Use forward_features directly
        output = self.model.forward_features(input)
        
        # For regression, we are interested in the output value itself
        # We don't need to select a specific class index
        
        # Compute gradients w.r.t the prediction
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        # Add alpha parameter required by relprop
        kwargs = {"alpha": 1}
        
        # Return attribution based on method
        # For regression, we use 1.0 as the starting relevance (not one-hot)
        return self.model.relprop(
            torch.ones_like(output).to(input.device), 
            method=method, 
            is_ablation=is_ablation, 
            start_layer=start_layer, 
            **kwargs
        )