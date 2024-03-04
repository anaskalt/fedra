import sys
import os
import torch
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net import Net
from utils.data_loader import DataLoaderHandler
from utils.weight_utils import WeightManager

class TestAvg(unittest.TestCase):

    def setUp(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_path = '../data/cell_data.csv'
        self.input_dim = 96
        self.epochs = 10
        self.rounds = 5

    def train_model(self, which_cell):
        data_loader_handler = DataLoaderHandler(self.csv_path, which_cell=which_cell)
        trainloader, testloader = data_loader_handler.load_data()
        net = Net(input_dim=self.input_dim).to(self.DEVICE)
        data_loader_handler.train(net=net, trainloader=trainloader, epochs=self.epochs, device=self.DEVICE)
        return net, testloader

    def test_model_training_and_averaging(self):
        # Train two models on different subsets of data
        net_0, testloader_0 = self.train_model(which_cell=0)
        net_1, testloader_1 = self.train_model(which_cell=1)

        # Test individual models before averaging
        loss_0, accuracy_0 = DataLoaderHandler(self.csv_path).test(net_0, testloader_0, self.DEVICE)
        loss_1, accuracy_1 = DataLoaderHandler(self.csv_path).test(net_1, testloader_1, self.DEVICE)

        print(f"Model 0 - Loss: {loss_0}, Accuracy: {accuracy_0}")
        print(f"Model 1 - Loss: {loss_1}, Accuracy: {accuracy_1}")

        # Perform FedAvg
        num_data_points = [len(testloader_0.dataset), len(testloader_1.dataset)]
        averaged_weights = WeightManager.average_weights([net_0.state_dict(), net_1.state_dict()], num_data_points)
        
        # Apply averaged weights to a new model
        averaged_model = Net(input_dim=self.input_dim).to(self.DEVICE)
        averaged_model.load_state_dict(averaged_weights)

        # Test averaged model
        loss_avg, accuracy_avg = DataLoaderHandler(self.csv_path).test(averaged_model, testloader_0, self.DEVICE)
        print(f"Averaged Model - Loss: {loss_avg}, Accuracy: {accuracy_avg}")

        self.assertIsNotNone(loss_avg)
        self.assertIsNotNone(accuracy_avg)

if __name__ == '__main__':
    unittest.main()
