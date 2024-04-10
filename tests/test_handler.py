#!/usr/bin/env python3

"""
Instructions for Running the Script in Multiple Terminals:

Before you run the test_handler.py script in each terminal, you'll need to set a unique KEY_PATH
for each instance of the script. This ensures each peer has a distinct identity in the network.

Here's how to set the KEY_PATH in each terminal session:

Terminal 1:
-----------
export KEY_PATH=node1.key  # Set a unique key path for the first peer
python test_handler.py     # Run the script for the first peer

Terminal 2:
-----------
export KEY_PATH=node2.key  # Set a unique key path for the second peer
python test_handler.py     # Run the script for the second peer

Terminal 3:
-----------
export KEY_PATH=node3.key  # Set a unique key path for the third peer
python test_handler.py     # Run the script for the third peer

This comment block provides guidance on how to execute multiple instances of the script,
simulating a network of peers with unique identities. Adjust the KEY_PATH and run the script in separate terminals to simulate multiple peers joining the P2P network.
"""
import sys
import os
# Adjust the path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from fedra.network.handler import P2PHandler
from fedra.utils.state import Status
import torch

# Configuration values for the P2P network
BOOTNODES = ["/ip4/127.0.0.1/udp/5000/quic-v1/p2p/12D3KooWSYoEJBh6UtfAT8wdepcvH2sjVGUrSFjgsofZwvNWgFPe"]
KEY_PATH = os.getenv("KEY_PATH", "default_node.key")  # Use environment variable for unique key paths
TOPIC = "model-net"
PACKET_SIZE = 1024

async def run_peer():
    # Initialize the P2PHandler with network configurations
    p2p_handler = P2PHandler(BOOTNODES, KEY_PATH, TOPIC, PACKET_SIZE)
    await p2p_handler.init_network()

    # Wait for a minimum number of peers to be connected
    await p2p_handler.wait_for_peers(min_peers=3)

    # Start listening to network messages
    asyncio.create_task(p2p_handler.start_listening())

    ### Publish a 'hello' message to announce presence to other peers
    #await p2p_handler.publish_hello()

    # Update and publish the peer's status
    p2p_handler.local_peer_status.status = Status.NONE
    for _ in range(3):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)
    await asyncio.sleep(10)

    p2p_handler.local_peer_status.status = Status.JOINED
    for _ in range(3):
        await p2p_handler.publish_status(p2p_handler.local_peer_status)

    await asyncio.sleep(10)

    # Generate and publish random weights for this peer
    dummy_weights = {f'layer_{i}': torch.randn(1, 10).tolist() for i in range(5)}
    p2p_handler.local_peer_weights.weights = dummy_weights
    for _ in range(3):
        await p2p_handler.publish_weights(p2p_handler.local_peer_weights)

    # Use sleep to allow time for message exchange, adjust the time as necessary
    await asyncio.sleep(10)

    # At the end, get and print the network summary
    network_summary = p2p_handler.network_state.get_network_summary()
    print("Network Summary:", network_summary)

    # Keep the script running to listen for incoming messages
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(run_peer())