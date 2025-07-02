import torch
import numpy as np

class RegressionLRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, method="transformer_attribution", is_ablation=False, start_layer=0):
        """
        Generate LRP relevance for a regression model.
        
        Args:
            input: Model input
            method: Attribution method
            is_ablation: Flag for ablation studies
            start_layer: Starting layer for rollout
            
        Returns:
            Relevance map
        """
        # Forward pass with gradient tracking
        input.requires_grad_(True)
        output = self.model(input)
        
        # For regression, we want to compute gradients w.r.t. the single output value
        # This is the key fix: we need to ensure the scalar output for backward pass
        if output.dim() > 1:
            output_scalar = output.sum()  # Sum if multiple outputs
        else:
            output_scalar = output
        
        # Create output vector for relprop (should match model output dimension)
        if output.dim() > 1:
            output_vector = torch.ones_like(output, dtype=torch.float32, device=output.device)
        else:
            output_vector = torch.ones((1, 1), dtype=torch.float32, device=output.device)
        
        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward(retain_graph=True)
        
        # Call relprop with our output vector
        return self.model.relprop(output_vector, method=method, 
                                  is_ablation=is_ablation,
                                  start_layer=start_layer, alpha=1)