# main.py
import asyncio
import torch
import configparser
from models.net import Net
from utils.data_loader import DataLoaderHandler
from p2p.p2p_handler import P2PHandler
from p2p.p2p_callbacks import message_callback

async def main():

    # Load configuration
    config = configparser.ConfigParser()
    config.read('config/node.conf')

    # Initialize and train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use DataLoaderHandler to load data
    data_loader_handler = DataLoaderHandler(
        csv_path=config['DEFAULT']['csv_path'],
        which_cell=int(config['DEFAULT']['which_cell']),
        window_len=int(config['DEFAULT']['input_dim']),
        batch_size=int(config['DEFAULT']['batch_size'])
    )

    #data_loader_handler = DataLoaderHandler(csv_path='./data/cell_data.csv', which_cell=0)
    train_loader, _ = data_loader_handler.load_data()

    #net = Net(input_dim=96).to(DEVICE)
    net = Net(input_dim=int(config['DEFAULT']['input_dim'])).to(DEVICE)
    # ... Training logic ...

    # Setup P2P network
    p2p_handler = P2PHandler(
        bootnodes=[config['P2P']['bootnodes']],
        key_path=config['P2P']['key_path'],
        topic=config['P2P']['topic']
    )
    #p2p_handler = P2PHandler(bootnodes=["..."], key_path="node.key", topic="model-net")
    await p2p_handler.init_network()

    # Publish weights and start the monitoring task
    await p2p_handler.publish_weights(net)
    monitoring_task = asyncio.create_task(p2p_handler.monitor_peers_and_weights(net))

    # Subscribe to weights with a custom callback
    message_callback_with_tracking = lambda message: message_callback(net, message, p2p_handler)
    await p2p_handler.subscribe_to_weights(message_callback_with_tracking)

    # Additional code as needed...
    await asyncio.Future()  # This will keep the main function running indefinitely

if __name__ == "__main__":
    asyncio.run(main())
