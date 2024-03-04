# utils/weight_utils.py
import io
import torch

def serialize_model_weights(model, chunk_size=None):
    """Serialize model weights to a byte array."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    serialized_data = buffer.read()

    if chunk_size is not None:
        return [serialized_data[i:i + chunk_size] for i in range(0, len(serialized_data), chunk_size)]
    return [serialized_data]

def deserialize_model_weights(weight_chunks, model):
    """Deserialize byte array chunks into model weights."""
    weights_bytes = b''.join(weight_chunks)
    buffer = io.BytesIO(weights_bytes)
    state_dict = torch.load(buffer)
    model.load_state_dict(state_dict)
    return model
