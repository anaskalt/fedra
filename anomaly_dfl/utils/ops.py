
"""
Provides utilities for model weight manipulation in Federated Learning scenarios.

This module offers functionalities for serializing and deserializing neural network
weights, which facilitates their transmission over networks in a federated learning
environment. Additionally, it implements Federated Averaging (FedAvg), a fundamental
algorithm for aggregating model updates from multiple clients.

Classes:
    WeightManager: Manages serialization, deserialization, and averaging of model weights.

Example:
    weight_manager = WeightManager(model)
    serialized_weights = weight_manager.serialize_weights()
    deserialized_model = weight_manager.deserialize_weights(serialized_weights)
    averaged_model = WeightManager.average_weights([model1.state_dict(), model2.state_dict()])
"""

import io
import torch

class WeightManager:
    """
    Handles serialization, deserialization, and federated averaging of model weights.

    Attributes:
        model (torch.nn.Module): Neural network model whose weights are managed.
    """

    def __init__(self, model):
        """
        Initializes WeightManager with a neural network model.

        Args:
            model (torch.nn.Module): The model to manage weights for.
        """
        self.model = model

    def serialize_weights(self, chunk_size=None):
        """
        Serializes the model's weights to a list of byte arrays.

        Args:
            chunk_size (int, optional): Size of chunks for the serialized data. Defaults to None.

        Returns:
            List[bytes]: Serialized weights split into chunks if chunk_size is specified.
        """
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        serialized_data = buffer.read()

        if chunk_size is not None:
            # Split the serialized data into chunks if chunk_size is specified
            return [serialized_data[i:i + chunk_size] for i in range(0, len(serialized_data), chunk_size)]
        return [serialized_data]

    def deserialize_weights(self, weight_chunks):
        """
        Deserializes byte array chunks into model weights and updates the model.

        Args:
            weight_chunks (List[bytes]): Serialized model weights in chunks.

        Returns:
            torch.nn.Module: The model updated with the deserialized weights.
        """
        weights_bytes = b''.join(weight_chunks)
        buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(buffer)
        self.model.load_state_dict(state_dict)
        return self.model

    @staticmethod
    def average_weights(state_dicts, num_data_points=None):
        """
        Averages multiple state dictionaries using the Federated Averaging (FedAvg) technique.

        Args:
            state_dicts (List[dict]): List of state dictionaries from different models.
            num_data_points (List[int], optional): Number of data points used for training each model. Defaults to None.

        Returns:
            dict: A state dictionary representing the averaged model weights.
        """
        avg_state_dict = {}
        total_data_points = sum(num_data_points) if num_data_points else len(state_dicts)

        for key in state_dicts[0]:
            if num_data_points:
                # Compute the weighted sum of weights for each key if num_data_points is provided
                weighted_sum = sum(state_dict[key] * (n / total_data_points) for state_dict, n in zip(state_dicts, num_data_points))
            else:
                # Perform simple averaging if num_data_points is not provided
                weighted_sum = sum(state_dict[key] for state_dict in state_dicts) / len(state_dicts)
            avg_state_dict[key] = weighted_sum

        return avg_state_dict
