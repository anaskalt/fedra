# main.py
import time
import asyncio
import torch
import configparser
from models.net import Net
from utils.data_loader import DataLoaderHandler
from p2p.p2p_handler import P2PHandler
from p2p.p2p_callbacks import message_callback_with_tracking, message_callback
from utils.weight_utils import WeightManager

async def main():
    # Load configuration
    config = configparser.ConfigParser()
    config.read('conf/node.conf')
    print("Configuration loaded.")

    # Initialize and train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DataLoaderHandler to load data
    data_loader_handler = DataLoaderHandler(
        csv_path=config['DEFAULT']['csv_path'],
        which_cell=int(config['DEFAULT']['which_cell']),
        window_len=int(config['DEFAULT']['input_dim']),
        batch_size=int(config['DEFAULT']['batch_size'])
    )
    print("DataLoaderHandler initialized.")

    # Setup P2P network
    p2p_handler = P2PHandler(
        bootnodes=config['P2P']['bootnodes'].split(','),
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic'],
        packet_size=int(config['P2P']['packet_size'])
    )
    print("Initializing P2P network...")
    await p2p_handler.init_network()
    print("P2P network initialized.")
    
    net = Net(input_dim=int(config['DEFAULT']['input_dim'])).to(DEVICE)

    # Create an asynchronous task for subscribing to weight messages
    asyncio.create_task(
        p2p_handler.subscribe_to_weights(lambda message: message_callback(net, message, p2p_handler))
    )
    print("Subscription task created.")

    # Publish a hello message
    print("Publishing hello message...")
    await p2p_handler.publish_hello_message()
    print("Hello message published.")

    rounds = int(config['DEFAULT']['rounds'])
    for round in range(rounds):
        print(f"Starting round {round + 1}/{rounds}")

        # Train the model
        train_loader, _ = data_loader_handler.load_data()
        net = Net(input_dim=int(config['DEFAULT']['input_dim'])).to(DEVICE)
        print("Training model...")
        data_loader_handler.train(net, train_loader, epochs=int(config['DEFAULT']['epochs']), device=DEVICE)
        print("Model training complete.")

        # Publish weights
        print("Publishing model weights...")
        await p2p_handler.publish_weights(net)
        print("Weights published.")

        # Monitor network and perform federated averaging
        start_time = time.time()
        while True:
            await asyncio.sleep(int(config['P2P']['update_interval']))
            current_peers = await p2p_handler.get_peers()
            p2p_handler.update_peer_statuses(current_peers)

            # Check if all active peers have sent their weights
            if all(peer_status.status == 'active' and peer_status.weights is not None for peer_status in p2p_handler.peer_statuses.values()):
                # Perform federated averaging
                print("Performing federated averaging...")
                weight_manager = WeightManager(net)
                state_dicts = [peer_status.weights for peer_status in p2p_handler.peer_statuses.values() if peer_status.weights]
                averaged_weights = WeightManager.average_weights(state_dicts)
                net.load_state_dict(averaged_weights)
                print("Federated averaging complete.")

                # Reset peer statuses for the next round
                p2p_handler.reset_peer_statuses()
                break

            # Exit the loop if the averaging timeout is reached
            if time.time() - start_time > int(config['P2P']['averaging_timeout']):
                print(f"Timeout reached in round {round + 1}, proceeding without averaging")
                p2p_handler.reset_peer_statuses()
                break

    # Keep the main function running indefinitely
    print("\nEntering indefinite run state.")
    await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
