
### TOBE REMOVED BEFORE PACKAGED
import sys

sys.path.insert(0,"../")

import os
import time
import asyncio
import torch
import configparser
import matplotlib.pyplot as plt

from anomaly_dfl.models.simple_nn_classification import Net
from anomaly_dfl.utils.process import DataLoaderHandler
from anomaly_dfl.utils.state import Status
from anomaly_dfl.network.handler import P2PHandler
from anomaly_dfl.utils.operations import Operations

### For debug purposes
ATTEMPTS = 3
torch.autograd.set_detect_anomaly(True)
# Ensure the metrics directory exists
METRICS_DIR = '../metrics'
os.makedirs(METRICS_DIR, exist_ok=True)

# Load configuration
def load_configuration():
    config = configparser.ConfigParser()
    config.read('../conf/node.conf')
    print("Configuration loaded.")
    return config

# Initialize DataLoaderHandler
def setup_data_loader(config):
    data_loader_handler = DataLoaderHandler(
        csv_path=config['DEFAULT']['csv_path'],
        which_cell=int(config['DEFAULT']['which_cell']),
        window_len=int(config['DEFAULT']['input_dim']),
        batch_size=int(config['DEFAULT']['batch_size'])
    )
    print("Model initialized.")
    return data_loader_handler

# Initialize model
def initialize_model(config):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(input_dim=int(config['DEFAULT']['input_dim'])).to(DEVICE)
    return net, DEVICE

# Setup P2P network (async)
async def setup_p2p_network(config):
    p2p_handler = P2PHandler(
        bootnodes=config['P2P']['bootnodes'].split(','),
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic'],
        packet_size=int(config['P2P']['packet_size'])
    )
    return p2p_handler

async def perform_federated_averaging(p2p_handler, net, DEVICE):
    if p2p_handler.network_state.check_state(Status.READY, Status.EXITED):
    #if  p2p_handler.network_state.check_state(Status.READY):
        averaged_weights = Operations.average_weights(p2p_handler.network_state)

        # Load the averaged weights into the model
        net.load_state_dict(averaged_weights)
        net.to(DEVICE)

        print("Federated averaging complete.")
        p2p_handler.local_peer_status.status = Status.COMPLETED
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

    else:
        print("Not all peers are ready for federated averaging.")

async def train_and_sync(p2p_handler, net, rounds, data_loader_handler, epochs, DEVICE, avg_timeout):

    training_losses = []
    testing_losses = []

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        # 1. Training
        #train_loader, _ = data_loader_handler.load_data()
        train_loader, test_loader = data_loader_handler.load_data()
        # Update status to TRAINING
        p2p_handler.local_peer_status.status = Status.TRAINING
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

        # Perform training and record loss
        training_loss = data_loader_handler.train(net, train_loader, epochs, DEVICE)
        training_losses.append(training_loss)
        print("Model training complete.")

        # Testing and recording test loss
        test_loss, _ = data_loader_handler.test(net, test_loader, DEVICE)
        testing_losses.append(test_loss)
        print("Model evaluation complete.")

        # 2. Publish the trained model's weights
        p2p_handler.local_peer_weights.weights = net.state_dict()
        p2p_handler.network_state.update_peer_weights(p2p_handler.local_peer_weights)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_weights(p2p_handler.local_peer_weights)
        await asyncio.sleep(10)

        print("Local model's weights published.")

        # 3. Update status to READY after publishing weights
        p2p_handler.local_peer_status.status = Status.READY
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

        print("Status updated to READY.")

        # 4. Wait for all peers to become READY before proceeding with federated averaging
        ready_wait_start = time.time()
        while not p2p_handler.network_state.check_state(Status.READY, Status.EXITED):
        #while not p2p_handler.network_state.check_state(Status.READY):
            await asyncio.sleep(2)
            if time.time() - ready_wait_start > avg_timeout:
                print("Timeout reached, proceeding with federated averaging.")
                break

        # 5. Perform federated averaging
        await perform_federated_averaging(p2p_handler, net, DEVICE)

        print(f"Round {round + 1} completed.")

    # After all rounds are complete, plot and save the training and testing losses
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(testing_losses, label='Testing Loss')
    plt.title('Training and Testing Losses per Round')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    loss_plot_filename = f'{METRICS_DIR}/losses_per_round_{p2p_handler.local_peer_status.peer_id}.png'

    plt.savefig(loss_plot_filename)
    #plt.show()

    print(f"Losses plot saved as {loss_plot_filename}.")

    p2p_handler.local_peer_status.status = Status.EXITED
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)
    print("Training and synchronization complete. Status updated to EXITED.")
    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)
    await asyncio.sleep(10)

async def main():
    config = load_configuration()
    rounds = int(config['DEFAULT']['rounds'])
    epochs = int(config['DEFAULT']['epochs'])
    avg_timeout = int(config['P2P']['averaging_timeout'])
    check_interval = int(config['P2P']['update_interval'])

    data_loader_handler = setup_data_loader(config)

    p2p_handler = await setup_p2p_network(config)

    await p2p_handler.init_network()

    # Wait for a minimum number of peers to be connected
    await p2p_handler.wait_for_peers(min_peers=1, check_interval=check_interval)

    net, DEVICE = initialize_model(config)

    # Subscribe to network messages and handle them appropriately
    asyncio.create_task(p2p_handler.subscribe_to_messages())

    # Update and publish the peer's status
    p2p_handler.local_peer_status.status = Status.NONE
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)
    await asyncio.sleep(10)

    p2p_handler.local_peer_status.status = Status.JOINED
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)
    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)

    await asyncio.sleep(10)

    # Training and synchronization loop
    await train_and_sync(p2p_handler, net, rounds, data_loader_handler, epochs, DEVICE, avg_timeout)

    #Keep the script running to listen for incoming messages
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())