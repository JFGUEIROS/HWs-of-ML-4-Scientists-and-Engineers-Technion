import torch
import torch.nn as nn

# Define neural network
class Question4(nn.Module):
    def __init__(self, in_dim=2, HL1_dim=2, out_dim=1):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(in_dim, HL1_dim), nn.Sigmoid(),
            nn.Linear(HL1_dim, out_dim), nn.Sigmoid()
        )

        # Initialize weights and biases for the first layer
        self.output[0].weight.data = torch.tensor([[0.1, 0.3], [0.2, 0.4]])  # W1
        self.output[0].bias.data = torch.tensor([0.5, 0.6])  # B1

        # Initialize weights and biases for the second layer
        self.output[2].weight.data = torch.tensor([[0.7, 0.8]])  # W2
        self.output[2].bias.data = torch.tensor([0.9])  # B2

    def forward(self, x):
        return self.output(x)

# Input tensor
x = torch.tensor([[0.3, 0.7]])  

# Target value
target = torch.tensor([[1.0]])  # Expected output

# Define model
model = Question4()

# Perform forward pass
output = model(x)

# Compute error using the given formula
error = 0.5 * ((output - target) ** 2).sum()  # Squared error

# Print output and error
print(f"Output: {output.item()}")
print(f"Error: {error.item()}")
