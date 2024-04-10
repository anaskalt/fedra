
"""
Contains the neural network architecture for anomaly detection tasks.

This module implements a straightforward, dense neural network using the PyTorch framework. 
Designed specifically for binary classification tasks such as anomaly detection, it features 
a simple architecture with one hidden layer. The network applies a ReLU activation function 
on the hidden layer and a sigmoid activation on the output layer, producing a probability 
indicative of anomaly presence.

Classes:
- Net: Defines the neural network architecture for anomaly detection.

Example:
    >>> import torch
    >>> from models.net import Net
    >>> model = Net(input_dim=100)
    >>> sample_input = torch.randn(1, 100)
    >>> output = model(sample_input)
    >>> print(output)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """
    Represents a dense neural network for anomaly detection.
    
    This class implements a neural network with one hidden layer for binary classification tasks, 
    particularly suited for detecting anomalies. The network consists of two linear layers, 
    leveraging ReLU and sigmoid activations for the hidden and output layers, respectively.

    Attributes:
        fc1 (torch.nn.Linear): The first fully connected layer.
        fc2 (torch.nn.Linear): The second fully connected layer, producing the output probability.

    Args:
        input_dim (int): The size of the input feature vector.

    Methods:
        forward(x): Performs the forward pass of the network.
    """

    def __init__(self, input_dim):
        """
        Initializes the Net with the specified input dimension.

        Args:
            input_dim (int): The size of the input feature vector.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, 1)

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): A tensor containing the input data.

        Returns:
            torch.Tensor: A tensor containing the output of the network.
        """
        x = F.relu(self.fc1(x))  # Apply ReLU to the first layer's output
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid to the second layer's output
        return x
