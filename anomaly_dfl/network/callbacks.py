
"""
Handles message callbacks in a peer-to-peer (P2P) federated learning network.

This module provides callback functions for processing messages received over the P2P network. 
It includes mechanisms for handling specific message types (e.g., 'hello', 'start', 'end') 
and managing the state of model weights during the federated learning process.

Functions:
    message_callback(net, message, p2p_handler): Processes incoming messages without specific peer tracking.
    message_callback_with_tracking(net, message, p2p_handler, peer_id): Processes messages with peer tracking.

Example:
    # Assuming existence of initialized `net`, `p2p_handler`, and a message `msg` received:
    message_callback(net, msg, p2p_handler)
"""

from anomaly_dfl.utils.operations import WeightManager
from anomaly_dfl.network.peer_status_manager import PeerStatus

def message_callback(net, message, p2p_handler):
    """
    Processes incoming messages and performs actions based on message content.

    This callback function handles general incoming P2P messages. It supports
    'hello from', 'start', and 'end' messages for initiating and tracking model
    weight updates over the network.

    Args:
        net: The neural network model.
        message (bytes): The received message.
        p2p_handler: The P2P network handler object.
    """
    if message.startswith(b'hello from'):
        # Handle hello message
        sender_peer_id = message.decode().split(" ")[-1]
        print(f"Received hello message from peer ID: {sender_peer_id}")

        # Ensure the sender_peer_id is in peer_statuses
        if sender_peer_id not in p2p_handler.peer_statuses:
            p2p_handler.peer_statuses[sender_peer_id] = PeerStatus(sender_peer_id)

        p2p_handler.current_sender_peer_id = sender_peer_id

    elif message == b'start':
        # Start of a new weight message
        p2p_handler.current_chunks = []

    elif message == b'end':
        # End of the current weight message
        weight_manager = WeightManager(net)
        updated_model = weight_manager.deserialize_weights(p2p_handler.current_chunks)
        net.load_state_dict(updated_model.state_dict())

        # Track the sender's peer ID for weight averaging
        if p2p_handler.current_sender_peer_id and p2p_handler.current_sender_peer_id in p2p_handler.peer_statuses:
            p2p_handler.peer_statuses[p2p_handler.current_sender_peer_id].weights = updated_model.state_dict()

        p2p_handler.current_chunks = None
        p2p_handler.current_sender_peer_id = None

    else:
        # Add chunk to current message
        if p2p_handler.current_chunks is not None:
            p2p_handler.current_chunks.append(message)

def message_callback_with_tracking(net, message, p2p_handler, peer_id):
    """
    Processes incoming messages with tracking of sender peer ID.

    Similar to `message_callback` but also tracks the peer ID that sent each message,
    allowing for more granular control over the federated learning process. It handles
    'start' and 'end' messages for managing model weight updates with peer tracking.

    Args:
        net: The neural network model.
        message (bytes): The received message.
        p2p_handler: The P2P network handler object.
        peer_id: The ID of the peer that sent the message.
    """
    if message == b'start':
        p2p_handler.current_chunks = []
    elif message == b'end':
        weight_manager = WeightManager(net)
        updated_model = weight_manager.deserialize_weights(p2p_handler.current_chunks)
        net.load_state_dict(updated_model.state_dict())
        # Store the deserialized weights in the corresponding PeerStatus object
        if peer_id not in p2p_handler.peer_statuses:
            p2p_handler.peer_statuses[peer_id] = PeerStatus(peer_id)
        p2p_handler.peer_statuses[peer_id].weights = updated_model.state_dict()
        p2p_handler.current_chunks = None
    else:
        p2p_handler.current_chunks.append(message)
