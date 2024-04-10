
"""
Manages Peer-to-Peer (P2P) network operations for Federated Learning.

This module contains the P2PHandler class responsible for managing peer-to-peer network interactions
in a federated learning setup. It includes functionality for initializing the network, publishing and
subscribing to messages (including model weights), and maintaining a record of peer statuses.

Classes:
    P2PHandler: Manages the P2P network interactions.

Example:
    # Initialization with network parameters
    p2p_handler = P2PHandler(bootnodes, key_path, topic, packet_size)
    asyncio.run(p2p_handler.init_network())
"""

from typing import List
import asyncio
import libp2p_pyrust as libp2p
from anomaly_dfl.utils.operations import Operations
from anomaly_dfl.utils.state import Status, PeerStatus, PeerWeights, NetworkState



class P2PHandler:
    """
    Handles initialization and communication over a P2P network for federated learning.

    Attributes:
        bootnodes (list): A list of bootnode addresses for network initialization.
        key_path (str): Path to the keypair used for node identification.
        topic (str): The network topic under which messages are published and subscribed.
        packet_size (int): The size of each packet for breaking down large messages, in bytes.
    """

    def __init__(self, bootnodes, key_path, topic, packet_size=1024):
        self.bootnodes = bootnodes
        self.key_path = key_path
        self.topic = topic
        self.packet_size = packet_size
        self.local_peer_id = None
        self.local_peer_status = None
        self.local_peer_weights = None
        self.network_state = NetworkState()
        self.message_buffer = bytearray()
        self.is_receiving = False

    def init_peer_objects(self, peer_id):
        """Initializes PeerStatus and PeerWeights with corresponding peer ID."""
        self.local_peer_status = PeerStatus(peer_id, Status.NONE)
        self.network_state.update_peer_status(self.local_peer_status)

        self.local_peer_weights = PeerWeights(peer_id, {})
        self.network_state.update_peer_weights(self.local_peer_weights)

    async def generate_or_load_key(self):
        """
        Generate or load a node keypair for P2P identity.
        Attempts to generate a keypair at the specified path; does nothing if it already exists.
        """
        try:
            libp2p.generate_ed25519_keypair(self.key_path)
        except FileExistsError:
            pass

    async def init_network(self):
        """
        Initialize the P2P network with the specified bootnodes and topic.
        Calls generate_or_load_key to ensure node identity before network initialization.
        """
        await self.generate_or_load_key()
        await libp2p.init_global_p2p_network(self.bootnodes, 0, self.key_path, self.topic)

        self.local_peer_id = await self.get_local_peer_id()
        self.init_peer_objects(self.local_peer_id)

    async def publish_hello(self):
        """
        Publishes a 'hello' message from the local peer to the network.
        This method is useful for announcing the peer's presence to other peers and can be
        extended to include additional metadata about the peer if needed.

        The method handles errors during message publishing and logs success or failure accordingly.
        """
        local_peer_id = await self.get_local_peer_id()
        hello_message = f"hello from {local_peer_id}".encode()

        try:
            await libp2p.publish_message(hello_message)
            print(f"Hello message published from {local_peer_id}.")
        except Exception as e:
            print(f"Failed to publish hello message from {local_peer_id}. Error: {str(e)}")

    async def publish_status(self, peer_status: PeerStatus):
        """
        Publishes the peer's status to the network. This function serializes the PeerStatus object
        and publishes it, allowing other peers in the network to be aware of the local peer's status.

        Args:
            peer_status (PeerStatus): The status object of the local peer.
        """
        operations = Operations(chunk_size=self.packet_size)
        serialized_data = operations.serialize(peer_status)
        await self.publish_objects(serialized_data)

    async def publish_weights(self, peer_weights: PeerWeights):
        """
        Publishes the peer's model weights to the network. This function serializes the PeerWeights object
        and publishes it, enabling other peers in the network to receive and utilize these weights for federated learning.

        Args:
            peer_weights (PeerWeights): The weights object of the local peer's model.
        """
        operations = Operations(chunk_size=self.packet_size)
        serialized_data = operations.serialize(peer_weights)
        await self.publish_objects(serialized_data)

    async def publish_objects(self, serialized_data: List[bytes], delay: float = 0.01):
        """
        Publishes serialized data objects (PeerWeights or PeerStatus) to the P2P network.

        This method serializes the data object (either PeerWeights or PeerStatus) into chunks
        and publishes each chunk to the network, with 'start' and 'end' flags to denote the sequence.
        It introduces an optional delay between sending chunks to prevent network congestion.

        Args:
            data_object (Union[PeerWeights, PeerStatus]): The data object to be serialized and published.
            delay (float): Delay in seconds between publishing each chunk to prevent network congestion. Defaults to 0.01.
        """

        await libp2p.publish_message(b'start')
        for chunk in serialized_data:
            await libp2p.publish_message(chunk)
            await asyncio.sleep(delay)
        await libp2p.publish_message(b'end')

    async def subscribe_to_messages(self):
        def callback_wrapper(message):
            self.message_dispatcher(message)
        
        await libp2p.subscribe_to_messages(callback_wrapper)


    async def start_listening(self):
        """
        Subscribes to messages from the P2P network and dispatches them to the appropriate handlers.
        This method remains effectively unchanged because the modifications to subscribe_to_messages
        ensure that asynchronous message dispatching is handled correctly.
        """
        await self.subscribe_to_messages()

    def message_dispatcher(self, message: bytes):
        """
        Dispatches incoming messages based on their type and content. It handles the start and end of message sequences
        and delegates processing of complete messages.

        Args:
            message (bytes): The received serialized message.
        """
        if message == b'start':
            self.message_buffer.clear()
            self.receiving = True
        elif message == b'end' and self.receiving:
            self.process_complete_message()
            self.receiving = False
        elif self.receiving:
            self.message_buffer.extend(message)

    def process_complete_message(self):
        """
        Processes a complete message after it's fully received, from the initial 'start' signal to the 'end' signal. 
        This involves deserializing the message into either a PeerStatus or PeerWeights object and updating the 
        network state accordingly. The method ensures accurate reflection of peer statuses and weights in the 
        network's overall state based on incoming data.
        """
        operations = Operations()
        data_object = operations.deserialize([self.message_buffer])
        if isinstance(data_object, PeerStatus):
            self.handle_peer_status_update(data_object)
        elif isinstance(data_object, PeerWeights):
            self.handle_peer_weights_update(data_object)
        self.message_buffer.clear()

    def handle_peer_status_update(self, peer_status: PeerStatus):
        """
        Handles the update of a peer's status. After a peer status message is fully received and processed, 
        this function updates the network state with the new status of the peer. It's crucial for maintaining 
        the current state of each peer within the network, facilitating coordinated actions and decisions.
        """
        self.network_state.update_peer_status(peer_status)
        print(f"PeerStatus updated for {peer_status.peer_id}: {peer_status.status}")

    def handle_peer_weights_update(self, peer_weights: PeerWeights):
        """
        Manages the update of a peer's model weights. Upon fully receiving and processing a peer weights message, 
        this function updates the network state with the new weights of the peer. This action is vital for the 
        federated learning process, allowing the network to collectively improve and refine the shared model based on peer contributions.
        """
        self.network_state.update_peer_weights(peer_weights)
        print(f"PeerWeights updated for {peer_weights.peer_id}.")

    async def get_peers(self):
        """
        Retrieves the list of currently connected peers.

        Returns:
            list: A list of peer IDs for all connected peers.
        """
        return await libp2p.get_global_connected_peers()

    async def get_local_peer_id(self):
        """
        Retrieves the local peer ID.

        Returns:
            str: The local peer ID.
        """
        return await libp2p.get_peer_id()

    async def wait_for_peers(self, min_peers=2, check_interval=5):
        """
        Waits for a minimum number of peers to be connected before proceeding.

        Args:
            min_peers (int): The minimum number of peers required to proceed.
            check_interval (int): The interval, in seconds, between checks for connected peers.
        Continuously checks for the number of connected peers and updates their statuses.
        Proceeds once the minimum number of peers is connected.
        """
        print(f"Waiting for at least {min_peers} peers to connect...")
        while True:
            current_peers = await self.get_peers()
            if len(current_peers) >= min_peers:
                print(f"Connected peers: {len(current_peers)}. Proceeding.")   
                break
            else:
                print(f"Connected peers: {len(current_peers)}. Waiting...")
                await asyncio.sleep(check_interval)
