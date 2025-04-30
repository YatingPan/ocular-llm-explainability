import torch

class RegressionLRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, method="transformer_attribution", is_ablation=False, start_layer=0):
        """
        Generate LRP relevance for a regression model, following a similar pattern to classification.
        
        Args:
            input: Model input
            method: Attribution method
            is_ablation: Flag for ablation studies
            start_layer: Starting layer for rollout
            
        Returns:
            Relevance map
        """
        # Forward pass
        output = self.model(input)
        
        # For regression, we create a similar one-hot-like vector
        # But it just contains a 1 since we only have one output
        output_vector = torch.ones((1, output.size()[-1]), dtype=torch.float32, device=output.device)
        
        # Similar to the classification approach, use this vector for backward pass
        one_hot = torch.sum(output_vector * output)
        
        # Compute gradients
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        # Call relprop with our output vector
        return self.model.relprop(output_vector, method=method, 
                                  is_ablation=is_ablation,
                                  start_layer=start_layer, alpha=1)