# utils/weight_utils.py
import io
import torch

def serialize_model_weights(model):
    """Serialize model weights to a byte array."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    return buffer.read()

def deserialize_model_weights(weights_bytes, model):
    """Deserialize byte array into model weights."""
    buffer = io.BytesIO(weights_bytes)
    state_dict = torch.load(buffer)
    model.load_state_dict(state_dict)
    return model
