# p2p/p2p_callbacks.py
from utils.weight_utils import deserialize_model_weights

def message_callback(net, message):
    """Handle incoming messages (model weights)."""
    updated_model = deserialize_model_weights(message, net)
    # Update model with new weights
    net.load_state_dict(updated_model.state_dict())

def message_callback_with_tracking(net, message, p2p_handler):
    """Handle incoming messages and track received weights."""
    updated_model = deserialize_model_weights(message, net)
    net.load_state_dict(updated_model.state_dict())
    p2p_handler.received_weights.add(message.sender_peer_id)
