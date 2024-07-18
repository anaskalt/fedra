
### TOBE REMOVED BEFORE PACKAGED
import sys

sys.path.insert(0,"../")

import os
import time
import asyncio
import torch
import configparser
import matplotlib.pyplot as plt

from fedra.models.simple_nn_classification import Net as SimpleNet
from fedra.models.dropout_lstm_regression import LSTMRegressor
from fedra.utils.process import DataLoaderHandler
from fedra.utils.state import Status
from fedra.network.handler import P2PHandler
from fedra.utils.operations import Operations

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

def get_model_configs(config):
    models = config['MODELS']['models'].split(', ')
    model_configs = {}
    
    for model in models:
        if model.upper() in config:
            model_config = dict(config[model.upper()])
            for key, value in model_config.items():
                if value.isdigit():
                    model_config[key] = int(value)
                elif value.replace('.', '').isdigit():
                    model_config[key] = float(value)
            model_configs[model] = model_config
        else:
            print(f"Warning: Configuration for model '{model}' not found.")
    
    return model_configs

# Initialize DataLoaderHandler
def setup_data_loader(model_config, model_type):
    # Helper function to safely convert to int
    def safe_int(value, default=30):
        try:
            # Strip any comments and convert to int
            return int(str(value).split('#')[0].strip())
        except (ValueError, TypeError):
            return default

    data_loader_handler = DataLoaderHandler(
        csv_path=model_config['dataset'],
        model_type=model_type,
        which_cell=model_config.get('which_cell', 0),
        window_len=safe_int(model_config.get('window_len', model_config.get('input_dim', 96))),
        batch_size=safe_int(model_config['batch_size']),
        grid_export_column=model_config.get('grid_export_column', 'DE_KN_residential3_grid_export'),
        training_days=safe_int(model_config.get('training_days', 30))
    )
    print(f"{model_type.upper()} DataLoader initialized with {model_config.get('grid_export_column', 'DE_KN_residential3_grid_export')} data.")
    return data_loader_handler

# Initialize model
def initialize_model(model_config, model_type):
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if model_type.startswith('simple_nn'):
        net = SimpleNet(input_dim=model_config['input_dim']).to(DEVICE)
    elif model_type.startswith('lstm'):
        net = LSTMRegressor(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout_rate=model_config['dropout_rate']
        ).to(DEVICE)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"{model_type.upper()} model initialized.")
    return net, DEVICE

# Setup P2P network (async)
async def setup_p2p_network(config):
    p2p_handler = P2PHandler(
        bootnodes=config['P2P']['bootnodes'].split(','),
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic'],
        packet_size=int(config['P2P']['packet_size']),
        min_peers=int(config['P2P']['min_peers'])
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

async def train_and_sync(p2p_handler, net, rounds, data_loader_handler, epochs, DEVICE, avg_timeout, model_type):
    training_losses = []
    testing_losses = []

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        train_loader, test_loader = data_loader_handler.load_data()
        p2p_handler.local_peer_status.status = Status.TRAINING
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

        training_loss = data_loader_handler.train(net, train_loader, epochs, DEVICE)
        training_losses.append(training_loss)
        print("Model training complete.")

        test_loss, _ = data_loader_handler.test(net, test_loader, DEVICE)
        testing_losses.append(test_loss)
        print("Model evaluation complete.")

        p2p_handler.local_peer_weights.weights = net.state_dict()
        p2p_handler.network_state.update_peer_weights(p2p_handler.local_peer_weights)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_weights(p2p_handler.local_peer_weights)
        await asyncio.sleep(10)

        print("Local model's weights published.")

        p2p_handler.local_peer_status.status = Status.READY
        p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)

        for _ in range(ATTEMPTS):
            await p2p_handler.publish_status(p2p_handler.local_peer_status)
        await asyncio.sleep(10)

        print("Status updated to READY.")

        ready_wait_start = time.time()
        while not p2p_handler.network_state.check_state(Status.READY, Status.EXITED):
            await asyncio.sleep(2)
            if time.time() - ready_wait_start > avg_timeout:
                print("Timeout reached, proceeding with federated averaging.")
                break

        await perform_federated_averaging(p2p_handler, net, DEVICE)

        print(f"Round {round + 1} completed.")

    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(testing_losses, label='Testing Loss')
    plt.title(f'Training and Testing Losses per Round ({model_type.upper()})')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.legend()

    loss_plot_filename = f'{METRICS_DIR}/losses_per_round_{model_type}_{p2p_handler.local_peer_status.peer_id}.png'
    plt.savefig(loss_plot_filename)
    print(f"Losses plot saved as {loss_plot_filename}.")

    p2p_handler.local_peer_status.status = Status.EXITED
    p2p_handler.network_state.update_peer_status(p2p_handler.local_peer_status)
    print("Training and synchronization complete. Status updated to EXITED.")
    for _ in range(ATTEMPTS):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)
    await asyncio.sleep(10)

async def main():
    config = load_configuration()
    model_configs = get_model_configs(config)
    avg_timeout = int(config['P2P']['averaging_timeout'])
    check_interval = int(config['P2P']['update_interval'])
    rounds = int(config['P2P']['rounds'])

    p2p_handler = await setup_p2p_network(config)
    await p2p_handler.init_network()
    await p2p_handler.wait_for_peers(check_interval=check_interval)

    asyncio.create_task(p2p_handler.subscribe_to_messages())

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

    for model_type, model_config in model_configs.items():
        print(f"Starting training for {model_type.upper()} model")
        data_loader_handler = setup_data_loader(model_config, model_type)
        net, DEVICE = initialize_model(model_config, model_type)
        
        epochs = model_config['epochs']

        await train_and_sync(p2p_handler, net, rounds, data_loader_handler, epochs, DEVICE, avg_timeout, model_type)

    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())