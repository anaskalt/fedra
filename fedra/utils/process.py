"""
Facilitates loading, preprocessing, and batching of datasets for anomaly detection and time series regression.

This module provides functionality to load datasets from CSV files, preprocess the data,
split it into training and testing sets, and encapsulate it into PyTorch DataLoader objects
for efficient training and testing of neural network models, including simple NN and LSTM models.

Classes:
    DataLoaderHandler: Loads and preprocesses dataset, provides DataLoader objects.

Example:
    data_loader_handler = DataLoaderHandler(csv_path='path/to/dataset.csv', model_type='lstm')
    train_loader, test_loader = data_loader_handler.load_data()
    # Use train_loader and test_loader with a neural network model
"""

import numpy as np
import pandas as pd
from functools import reduce
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DataLoaderHandler:
    """
    Handles loading, preprocessing, and batching of datasets for anomaly detection and time series regression.

    Attributes:
        csv_path (str): Path to the CSV dataset file.
        model_type (str): Type of model to preprocess for ('nn' or 'lstm').
        which_cell (int): Cell number to focus on in the dataset (for 'nn' type).
        window_len (int): Length of the sliding window for input features.
        batch_size (int): Batch size for training and testing DataLoaders.
        test_size (float): Proportion of the dataset to include in the test split.

    Methods:
        load_data: Loads dataset, preprocesses, and splits into training and testing DataLoaders.
        _preprocess_nn: Preprocesses data for simple NN model.
        _preprocess_lstm: Preprocesses data for LSTM model.
        train: Trains a neural network model using the training DataLoader.
        test: Evaluates a neural network model using the testing DataLoader.
    """

    def __init__(self, csv_path, model_type='nn', which_cell=0, window_len=96, batch_size=32, test_size=0.2, grid_export_column='DE_KN_residential3_grid_export', training_days=None):
        """
        Initializes DataLoaderHandler with dataset path and preprocessing parameters.

        Args:
            csv_path (str): Path to the CSV dataset file.
            model_type (str): Type of model to preprocess for ('nn' or 'lstm').
            which_cell (int): Cell number to target in the dataset (for 'nn' type).
            window_len (int): Length of the sliding window for input features.
            batch_size (int): Batch size for the DataLoaders.
            test_size (float): Proportion of the dataset to include in the test split.
        """
        self.csv_path = csv_path
        self.model_type = model_type
        self.which_cell = which_cell
        self.window_len = window_len
        self.batch_size = batch_size
        self.test_size = test_size
        self.grid_export_column = grid_export_column
        self.training_days = training_days

    def load_data(self):
        """
        Loads dataset from CSV, preprocesses it, and returns training and testing DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
        """
        if self.model_type.startswith('simple_nn'):
            X, y = self._preprocess_nn()
        elif self.model_type.startswith('lstm'):
            X, y = self._preprocess_lstm()
        else:
            raise ValueError(f"Invalid model_type: {self.model_type}. Choose a type starting with 'simple_nn' or 'lstm'.")

        # Split dataset and create DataLoaders
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size)

        return train_loader, test_loader

    def _preprocess_nn(self):
        """
        Preprocesses data for the simple NN model (anomaly detection).

        Returns:
            Tuple[np.array, np.array]: Preprocessed features and labels.
        """
        dataset_train = pd.read_csv(self.csv_path)
        train_data = dataset_train.values

        anomaly_data = np.zeros((5, 3, 5, 16608))
        load_data = np.zeros((5, 3, 5, 16608))

        for cell in range(5):
            for category in range(3):
                for device in range(5):
                    rows = reduce(np.intersect1d, (np.where(train_data[:, 1] == cell),
                                                   np.where(train_data[:, 2] == category),
                                                   np.where(train_data[:, 3] == device)))
                    load_data[cell, category, device] = train_data[rows, 4]
                    anomaly_data[cell, category, device] = train_data[rows, 5]

        load_set = np.sum(np.sum(load_data[self.which_cell, :, :, :], axis=0), axis=0)
        anomaly_set = np.sum(np.sum(anomaly_data[self.which_cell, :, :, :], axis=0), axis=0)
        anomaly_set[anomaly_set != 0] = 1  # Binary classification

        sc = MinMaxScaler(feature_range=(0, 1))
        load_set_scaled = sc.fit_transform(load_set.reshape(-1, 1))

        X, y = [], []
        for i in range(self.window_len, len(load_set_scaled)):
            X.append(load_set_scaled[i-self.window_len:i, 0])
            y.append(anomaly_set[i])
        
        return np.array(X), np.array(y)

    def _preprocess_lstm(self):
        """
        Preprocesses data for the LSTM model (time series regression).

        Returns:
            Tuple[np.array, np.array]: Preprocessed features and labels.
        """
        dataset_train = pd.read_csv(self.csv_path)
        
        if self.grid_export_column not in dataset_train.columns:
            raise ValueError(f"Column {self.grid_export_column} not found in the dataset.")

        train_data = dataset_train[self.grid_export_column]

        if self.training_days is None:
            self.training_days = len(train_data) // 24  # Assume hourly data
        else:
            self.training_days = int(self.training_days)  # Ensure it's an integer

        # Calculate differences directly on the pandas Series
        training_data = train_data.diff().dropna().values

        # Limit the data to the specified training days
        training_data = training_data[:self.training_days * 24]

        sc = MinMaxScaler(feature_range=(0, 1))
        training_set_scaled = sc.fit_transform(training_data.reshape(-1, 1))

        X, y = [], []
        for i in range(self.window_len, len(training_set_scaled)):
            X.append(training_set_scaled[i-self.window_len:i, 0])
            y.append(training_set_scaled[i, 0])
        
        X = np.array(X)
        y = np.array(y)

        # Reshape X to (samples, time steps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        return X, y

    def train(self, net, trainloader, epochs, device):
        """
        Trains a neural network model using the specified DataLoader.

        Args:
            net (nn.Module): Neural network model to train.
            trainloader (DataLoader): DataLoader for training data.
            epochs (int): Number of epochs to train for.
            device (torch.device): Device to train the model on.

        Returns:
            float: The average loss over all training epochs.
        """
        criterion = nn.BCELoss() if self.model_type == 'nn' else nn.MSELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)  # learning rate to be configurable
        net.to(device)
        net.train()
        
        total_loss = 0
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        average_loss = total_loss / (len(trainloader.dataset) * epochs)
        return average_loss

    def test(self, net, testloader, device):
        """
        Evaluates a neural network model using the specified DataLoader.

        Args:
            net (nn.Module): Neural network model to evaluate.
            testloader (DataLoader): DataLoader for testing data.
            device (torch.device): Device to evaluate the model on.

        Returns:
            Tuple[float, float]: A tuple containing the average loss and accuracy over the testing set.
        """
        criterion = nn.BCELoss() if self.model_type == 'nn' else nn.MSELoss()
        net.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                if self.model_type == 'nn':
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == labels).sum().item()
                total += labels.size(0)
        average_loss = total_loss / len(testloader.dataset)
        accuracy = correct / total if self.model_type == 'nn' else None
        return average_loss, accuracy
