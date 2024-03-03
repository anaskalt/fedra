# main.py
import asyncio
from models.net import Net
from utils.data_loader import load_data
from p2p.p2p_handler import P2PHandler
from p2p.p2p_callbacks import message_callback

async def main():
    # Initialize and train the model
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, _ = load_data('./data/cell_data.csv', which_cell=0)
    net = Net(input_dim=96).to(DEVICE)
    # ... Training logic ...

    # Setup P2P network
    p2p_handler = P2PHandler(bootnodes=["..."], key_path="node.key", topic="model-net")
    await p2p_handler.init_network()

    # Publish weights and start the monitoring task
    await p2p_handler.publish_weights(net)
    monitoring_task = asyncio.create_task(p2p_handler.monitor_peers_and_weights(net))

    # Subscribe to weights with a custom callback
    message_callback_with_tracking = lambda message: message_callback_with_tracking(net, message, p2p_handler)
    await p2p_handler.subscribe_to_weights(message_callback_with_tracking)

    # Additional code as needed...
    await asyncio.Future()  # This will keep the main function running indefinitely

if __name__ == "__main__":
    asyncio.run(main())