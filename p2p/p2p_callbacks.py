# p2p/p2p_callbacks.py
from utils.weight_utils import WeightManager
from p2p.peer_status_manager import PeerStatus

def message_callback(net, message, p2p_handler):
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
    """Handle incoming messages and track received weights."""
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