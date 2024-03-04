# models/net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Dense Neural Network for Anomaly Detection."""
    def __init__(self, input_dim) -> None:
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
