"""
Contains the LSTM neural network architecture for time series regression tasks.

This module implements a Long Short-Term Memory (LSTM) neural network using the PyTorch framework. 
Designed specifically for time series regression tasks such as energy consumption prediction, 
it features a stacked LSTM architecture with dropout layers. The network applies four LSTM layers 
followed by dropout, and a final linear layer for output prediction.

Classes:
- LSTMRegressor: Defines the LSTM neural network architecture for time series regression.

Example:
    >>> import torch
    >>> from models.dropout_lstm_regression import LSTMRegressor
    >>> model = LSTMRegressor(input_size=1, hidden_size=50, num_layers=4, output_size=1)
    >>> sample_input = torch.randn(1, 14*24, 1)  # Assuming 14 days of hourly data
    >>> output = model(sample_input)
    >>> print(output)
"""

import torch.nn as nn

class LSTMRegressor(nn.Module):
    """
    Represents a stacked LSTM neural network for time series regression.
    
    This class implements a neural network with four LSTM layers and dropout for regression tasks, 
    particularly suited for predicting time series data like energy consumption. The network 
    consists of four LSTM layers with dropout, followed by a final linear layer for prediction.

    Attributes:
        hidden_size (int): The number of features in the hidden state of the LSTM layers.
        num_layers (int): The number of LSTM layers stacked on each other.
        lstm1, lstm2, lstm3, lstm4 (nn.LSTM): The four LSTM layers.
        dropout1, dropout2, dropout3, dropout4 (nn.Dropout): Dropout layers after each LSTM.
        fc (nn.Linear): The final fully connected layer for output prediction.

    Args:
        input_size (int): The number of expected features in the input x.
        hidden_size (int): The number of features in the hidden state h.
        num_layers (int): The number of LSTM layers.
        output_size (int): The size of the output.
        dropout_rate (float): Probability of an element to be zeroed in dropout layers.

    Methods:
        forward(x): Performs the forward pass of the network.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        """
        Initializes the LSTMRegressor with the specified parameters.

        Args:
            input_size (int): The number of expected features in the input x.
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): The number of LSTM layers.
            output_size (int): The size of the output.
            dropout_rate (float): Probability of an element to be zeroed in dropout layers.
        """
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): A tensor containing the input data, expected shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: A tensor containing the output of the network, shape (batch_size, output_size).
        """
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        
        x = self.fc(x[:, -1, :])
        return x
