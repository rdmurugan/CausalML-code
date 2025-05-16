import torch

def wasserstein_distance(a, b):  # Simplified Wasserstein
    return torch.mean(a) - torch.mean(b)

def balancing_loss(representation, treatment):
    treated_representation = representation[treatment == 1]
    control_representation = representation[treatment == 0]
    return wasserstein_distance(treated_representation, control_representation)

# Example Usage
representation = torch.randn(100, 10)  # Example representation
treatment = torch.randint(0, 2, (100,))  # Example treatment (0 or 1)
treatment = treatment.float() # Ensure treatment is float for calculations
bal_loss = balancing_loss(representation, treatment)
print(f"Balancing Loss: {bal_loss.item()}")