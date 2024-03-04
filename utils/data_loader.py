# utils/data_loader.py
import numpy as np
import pandas as pd
from functools import reduce
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.net import Net

class DataLoaderHandler:
    def __init__(self, csv_path, which_cell=0, window_len=96, batch_size=32):
        self.csv_path = csv_path
        self.which_cell = which_cell
        self.window_len = window_len
        self.batch_size = batch_size

    def load_data(self):
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
        anomaly_set[anomaly_set != 0] = 1

        sc = MinMaxScaler(feature_range=(0, 1))
        load_set_scaled = sc.fit_transform(load_set.reshape(-1, 1))

        X, y = [], []
        for i in range(self.window_len, len(load_set_scaled)):
            X.append(load_set_scaled[i-self.window_len:i, 0])
            y.append(anomaly_set[i])
        X, y = np.array(X), np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))

        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=32)

        return train_loader, test_loader

    def train(self, net: Net, trainloader: DataLoader, epochs: int, device: torch.device) -> float:
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
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

    def test(self, net: Net, testloader: DataLoader, device: torch.device) -> Tuple[float, float]:
        criterion = nn.BCELoss()
        net.eval()
        
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        average_loss = total_loss / len(testloader.dataset)
        accuracy = correct / total
        return average_loss, accuracy