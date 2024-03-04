import sys
import os
import torch
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net import Net
from utils.data_loader import DataLoaderHandler
from utils.weight_utils import WeightManager

class TestAvgRounds(unittest.TestCase):

    def setUp(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.csv_path = '../data/cell_data.csv'
        self.input_dim = 96
        self.epochs = 10
        self.rounds = 5

    def train_model(self, net, which_cell):
        data_loader_handler = DataLoaderHandler(self.csv_path, which_cell=which_cell)
        trainloader, _ = data_loader_handler.load_data()
        data_loader_handler.train(net=net, trainloader=trainloader, epochs=self.epochs, device=self.DEVICE)
        return len(trainloader.dataset)

    def test_model_training_and_averaging(self):
        # Initialize two models
        net_0 = Net(input_dim=self.input_dim).to(self.DEVICE)
        net_1 = Net(input_dim=self.input_dim).to(self.DEVICE)

        for round in range(self.rounds):
            print(f"Round {round + 1}")

            # Train each model and get the number of data points
            num_data_points_0 = self.train_model(net_0, which_cell=0)
            num_data_points_1 = self.train_model(net_1, which_cell=1)

            # Average weights using FedAvg
            averaged_weights = WeightManager.average_weights([net_0.state_dict(), net_1.state_dict()], [num_data_points_0, num_data_points_1])

            # Load averaged weights into both models for the next round
            net_0.load_state_dict(averaged_weights)
            net_1.load_state_dict(averaged_weights)

        # Test averaged models after all rounds
        data_loader_handler_0 = DataLoaderHandler(self.csv_path, which_cell=0)
        _, testloader_0 = data_loader_handler_0.load_data()
        data_loader_handler_1 = DataLoaderHandler(self.csv_path, which_cell=1)
        _, testloader_1 = data_loader_handler_1.load_data()
        
        loss_0, accuracy_0 = data_loader_handler_0.test(net_0, testloader_0, self.DEVICE)
        loss_1, accuracy_1 = data_loader_handler_1.test(net_1, testloader_1, self.DEVICE)

        print(f"Model 0 after averaging - Loss: {loss_0}, Accuracy: {accuracy_0}")
        print(f"Model 1 after averaging - Loss: {loss_1}, Accuracy: {accuracy_1}")

        self.assertIsNotNone(loss_0)
        self.assertIsNotNone(accuracy_0)
        self.assertIsNotNone(loss_1)
        self.assertIsNotNone(accuracy_1)

if __name__ == '__main__':
    unittest.main()
