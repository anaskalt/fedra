# p2p/p2p_handler.py
import asyncio
import libp2p_pyrust as libp2p
from utils.weight_utils import WeightManager
from p2p.peer_status_manager import PeerStatus

class P2PHandler:
    #def __init__(self, bootnodes, key_path, topic, packet_size = 1024, update_interval = 30, averaging_timeout = 300):
    def __init__(self, bootnodes, key_path, topic, packet_size = 1024): #1024 byte per packet
        self.bootnodes = bootnodes
        self.key_path = key_path
        self.topic = topic
        self.packet_size = packet_size
        self.current_sender_peer_id = None
        #self.update_interval = update_interval
        #self.averaging_timeout = averaging_timeout
        self.peer_statuses = {}

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

    async def publish_hello_message(self, retry_interval=5, max_retries=5):
        """Publish a hello message with retries."""
        # Retrieve the local peer ID
        local_peer_id = await libp2p.get_peer_id()
        hello_message = f"hello from {local_peer_id}".encode()  # Encode the message to bytes

        for attempt in range(max_retries):
            try:
                await libp2p.publish_message(hello_message)
                print("Hello message published. OK")
                return
            except RuntimeError as e:
                if "InsufficientPeers" in str(e):
                    print(f"Retry {attempt + 1}/{max_retries}: Waiting for peers...")
                    await asyncio.sleep(retry_interval)
                else:
                    raise
        print("Failed to publish hello message after retries.")

    async def publish_weights(self, model):
        """Publish model weights with start and end flags."""
        weight_manager = WeightManager(model)
        serialized_weights = weight_manager.serialize_weights(chunk_size=self.packet_size)
        # Send start flag
        await libp2p.publish_message(b'start')
        # Send serialized weights in chunks
        for chunk in serialized_weights:
            await libp2p.publish_message(chunk)
        # Send end flag
        await libp2p.publish_message(b'end')

    '''async def publish_weights(self, model):
        """Publish model weights."""
        weights = serialize_model_weights(model)
        await libp2p.publish_message(list(weights))'''

    async def subscribe_to_weights(self, callback):
        """Subscribe to model weights."""
        await libp2p.subscribe_to_messages(callback)

    async def get_peers(self):
        """Retrieve the list of connected peers."""
        return await libp2p.get_global_connected_peers()

    '''async def monitor_peers_and_weights(self, model):
        start_time = time.time()
        while True:
            current_peers = await self.get_peers()
            self.update_peer_statuses(current_peers)

            if time.time() - start_time > self.averaging_timeout:
                if any(status.weights for status in self.peer_statuses.values()):
                    weight_manager = WeightManager(model)
                    averaged_weights = weight_manager.average_weights([status.weights for status in self.peer_statuses.values() if status.weights])
                    model.load_state_dict(averaged_weights)

                start_time = time.time()
                self.reset_peer_statuses()
            await asyncio.sleep(self.update_interval)'''

    def update_peer_statuses(self, current_peers):
        for peer_id in current_peers:
            if peer_id not in self.peer_statuses:
                self.peer_statuses[peer_id] = PeerStatus(peer_id)

            # Update peer status
            self.peer_statuses[peer_id].status = "active"

        # Mark missing peers as inactive
        for peer_id in self.peer_statuses:
            if peer_id not in current_peers:
                self.peer_statuses[peer_id].status = "inactive"

    def reset_peer_statuses(self):
        for peer_status in self.peer_statuses.values():
            peer_status.weights = None
            peer_status.status = "active"


''' async def monitor_peers(self):
        """Monitor for changes in peers."""
        while True:
            current_peers = await self.get_peers()
            for peer in current_peers:
                if peer not in self.peer_status.peers or not self.peer_status.peers[peer]['active']:
                    self.peer_status.update_peer(peer)

            # Check for peers that are no longer active
            for peer in self.peer_status.peers:
                if peer not in current_peers:
                    self.peer_status.remove_peer(peer)

            await asyncio.sleep(self.update_interval)

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

    async def monitor_peers_and_weights(self, model, update_interval=30, averaging_timeout=300):
        """Monitor for new peers and weights."""
        start_time = time.time()
        while True:
            current_peers = await self.get_peers()
            if time.time() - start_time > averaging_timeout:
                if self.received_weights:
                    # Perform federated averaging
                    weight_manager = WeightManager(model)
                    averaged_weights = weight_manager.average_weights(list(self.received_weights.values()))
                    model.load_state_dict(averaged_weights)
                start_time = time.time()
                self.received_weights.clear()
            await asyncio.sleep(update_interval)'''
