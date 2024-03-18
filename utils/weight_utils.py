# utils/weight_utils.py
import io
import torch

class WeightManager:
    def __init__(self, model):
        self.model = model

    def serialize_weights(self, chunk_size=None):
        """Serialize model weights to a byte array."""
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        serialized_data = buffer.read()

        if chunk_size is not None:
            return [serialized_data[i:i + chunk_size] for i in range(0, len(serialized_data), chunk_size)]
        return [serialized_data]

    def deserialize_weights(self, weight_chunks):
        """Deserialize byte array chunks into model weights."""
        weights_bytes = b''.join(weight_chunks)
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer)
        self.model.load_state_dict(state_dict)
        return self.model

    '''def average_weights(self, list_of_weights):
        """Average weights using the Federated Averaging (FedAvg) technique."""
        average_weights = {}
        num_weights = len(list_of_weights)

        # Initialize average_weights with zero tensors
        for key in list_of_weights[0].keys():
            average_weights[key] = torch.zeros_like(list_of_weights[0][key])

        # Sum up all weights
        for weights in list_of_weights:
            for key in weights.keys():
                average_weights[key] += weights[key]

        # Average the weights
        for key in average_weights.keys():
            average_weights[key] /= num_weights

        # Load averaged weights into model
        self.model.load_state_dict(average_weights)
        return self.model'''

    '''@staticmethod
    def average_weights(state_dicts, num_data_points=None):
        """Average multiple state dictionaries using FedAvg technique."""
        avg_state_dict = {}
        total_data_points = sum(num_data_points) if num_data_points else len(state_dicts)

        for key in state_dicts[0]:
            # Compute weighted sum of the weights for each key
            weighted_sum = sum(state_dict[key] * (n / total_data_points) for state_dict, n in zip(state_dicts, num_data_points))
            avg_state_dict[key] = weighted_sum

        return avg_state_dict'''

    @staticmethod
    def average_weights(state_dicts, num_data_points=None):
        """Average multiple state dictionaries using FedAvg technique."""
        avg_state_dict = {}
        total_data_points = sum(num_data_points) if num_data_points else len(state_dicts)

        for key in state_dicts[0]:
            if num_data_points:
                # Compute weighted sum of the weights for each key
                weighted_sum = sum(state_dict[key] * (n / total_data_points) for state_dict, n in zip(state_dicts, num_data_points))
            else:
                # Simple average if num_data_points is None
                weighted_sum = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)
            avg_state_dict[key] = weighted_sum

        return avg_state_dict