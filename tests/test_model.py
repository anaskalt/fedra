# tests/test_model.py
import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedra.models.simple_nn_classification import Net
from fedra.utils.process import DataLoaderHandler

def test_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = '../data/cell_data.csv'

    print("PyTorch model training")
    print("Load data")

    data_loader_handler = DataLoaderHandler(csv_path)
    trainloader, testloader = data_loader_handler.load_data()

    net = Net(input_dim=96).to(DEVICE)
    net.eval()

    print("Start training")
    data_loader_handler.train(net=net, trainloader=trainloader, epochs=100, device=DEVICE)

    print("Evaluate model")
    loss, accuracy = data_loader_handler.test(net=net, testloader=testloader, device=DEVICE)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)

if __name__ == "__main__":
    test_model()
