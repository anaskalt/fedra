# p2p/p2p_handler.py
import asyncio
import time
import libp2p_pyrust as libp2p
from utils.weight_utils import serialize_model_weights

class P2PHandler:
    def __init__(self, bootnodes, key_path, topic):
        self.bootnodes = bootnodes
        self.key_path = key_path
        self.topic = topic

    async def generate_or_load_key(self):
        """Generate or load the node keypair."""
        try:
            libp2p.generate_ed25519_keypair(self.key_path)
        except FileExistsError:
            pass

    async def init_network(self):
        """Initialize the P2P network."""
        await self.generate_or_load_key()
        await libp2p.init_global_p2p_network(self.bootnodes, 0, self.key_path, self.topic)

    async def publish_weights(self, model):
        """Publish model weights."""
        weights = serialize_model_weights(model)
        await libp2p.publish_message(list(weights))

    async def subscribe_to_weights(self, callback):
        """Subscribe to model weights."""
        await libp2p.subscribe_to_messages(callback)

    async def get_peers(self):
        """Retrieve the list of connected peers."""
        return await libp2p.get_global_connected_peers()

    async def monitor_peers_and_weights(self, net, update_interval=30, averaging_timeout=300):
        """Monitor for new peers and weights."""
        start_time = time.time()
        received_weights = set()
        while True:
            current_peers = await self.get_peers()
            if len(current_peers) == len(received_weights) or time.time() - start_time > averaging_timeout:
                if received_weights:
                    net.average_weights(received_weights)
                start_time = time.time()
                received_weights.clear()
            await asyncio.sleep(update_interval)
