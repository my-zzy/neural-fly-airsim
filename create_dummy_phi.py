import torch
import torch.nn as nn

# Create a simple phi network architecture for testing
class PhiNet(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=64, output_features=10):  # 11 inputs: vel(3) + quat(4) + rotors(4)
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_features * 3)  # 3 dimensions x output_features
        self.output_features = output_features
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.view(3, self.output_features)  # Reshape to [3, output_features]

# Create and save a dummy trained network
if __name__ == "__main__":
    phi_net = PhiNet()
    torch.save(phi_net, "trained_phi.pt")
    print("Created dummy trained_phi.pt file for testing")
