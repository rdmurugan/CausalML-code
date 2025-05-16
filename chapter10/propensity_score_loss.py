import torch
import torch.nn.functional as F

def propensity_loss(predicted_propensity, treatment):
    return F.binary_cross_entropy(predicted_propensity, treatment)

# Example
predicted_propensity = torch.sigmoid(torch.randn(100, 1))  # Probabilities
treatment = torch.randint(0, 2, (100, 1)).float()
prop_loss = propensity_loss(predicted_propensity, treatment)
print(f"Propensity Loss: {prop_loss.item()}")