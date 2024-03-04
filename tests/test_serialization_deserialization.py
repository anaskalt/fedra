import sys
import os
import torch
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.net import Net
from utils.data_loader import DataLoaderHandler
from utils.weight_utils import WeightManager

class TestSerializationDeserialization(unittest.TestCase):

    def setUp(self):
        print("Setting up the test environment...")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        csv_path = '../data/cell_data.csv'
        self.data_loader_handler = DataLoaderHandler(csv_path)
        self.input_dim = 96
        self.epochs = 10
        self.chunk_size = 1024  # Adjust as needed

    def test_serialization_deserialization(self):
        print("Loading data...")
        trainloader, _ = self.data_loader_handler.load_data()
        print("Initializing model...")
        net = Net(input_dim=self.input_dim).to(self.DEVICE)
        print("Training model...")
        self.data_loader_handler.train(net=net, trainloader=trainloader, epochs=self.epochs, device=self.DEVICE)

        print("Creating WeightManager and serializing model weights...")
        weight_manager = WeightManager(net)
        original_state_dict = net.state_dict()

        # Serialize and chunk the model weights
        serialized_weights = weight_manager.serialize_weights(self.chunk_size)
        #serialized_weights = serialize_model_weights(net, self.chunk_size)

        print("Deserializing model weights...")
        # Deserialize the weights and load back into the model
        deserialized_model = weight_manager.deserialize_weights(serialized_weights)
        #deserialized_model = deserialize_model_weights(serialized_weights, net)
        deserialized_state_dict = deserialized_model.state_dict()

        #print("Saving original and deserialized state dictionaries...")
        #torch.save(original_state_dict, 'original_state_dict.pth')
        #torch.save(deserialized_state_dict, 'deserialized_state_dict.pth')

        print("Checking if the state dictionaries are identical...")
        # Check if the state dictionaries are identical
        for key in original_state_dict:
            self.assertTrue(torch.equal(original_state_dict[key], deserialized_state_dict[key]))

if __name__ == '__main__':
    unittest.main()
