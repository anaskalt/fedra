#!/usr/bin/env python3

"""Test script for Operations functionality."""

import sys
import os

# Adjust the path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fedra.utils.state import PeerWeights, NetworkState
from fedra.utils.operations import Operations

def create_dummy_weights():
    """Creates dummy weights for simulation."""
    return {f'layer_{i}': torch.randn(1, 10) for i in range(5)}

def weights_equal(weights1, weights2):
    """Check if two sets of weights are equal."""
    for key in weights1.keys():
        if not torch.equal(weights1[key], weights2[key]):
            return False
    return True

def test_serialize_deserialize():
    print("Testing serialization and deserialization...")

    operations = Operations(chunk_size=1024)
    dummy_weights = create_dummy_weights()
    peer_weights = PeerWeights("peer1", dummy_weights)


    # Print data before serialization
    print("Data object before serialization:")
    print("Peer ID:", peer_weights.peer_id)
    print("Weights:")
    for key, value in peer_weights.weights.items():
        print(f"{key}: {value}")
    print()

    # Serialize
    serialized_data = operations.serialize(peer_weights)
    print("Serialized data:", [len(chunk) for chunk in serialized_data], "...truncated for readability")
    print()

    # Deserialize
    deserialized_peer_weights = operations.deserialize(serialized_data)
    print("Deserialized PeerWeights object:")
    print("Peer ID:", deserialized_peer_weights.peer_id)
    print("Weights:")
    for key, value in deserialized_peer_weights.weights.items():
        print(f"{key}: {value}")

    # Check if original and deserialized weights are equal
    has_diff = not weights_equal(peer_weights.weights, deserialized_peer_weights.weights)
    print(f"Data consistency check (should be False): {has_diff}\n")

def test_average_weights():
    print("\nTesting federated averaging...")

    network_state = NetworkState()
    operations = Operations()

    # Simulate adding peer weights to the network state
    for i in range(1, 4):
        dummy_weights = create_dummy_weights()
        peer_weights = PeerWeights(f"peer{i}", dummy_weights)
        network_state.update_peer_weights(peer_weights)

    # Federated averaging
    averaged_weights = operations.average_weights(network_state)
    print("Averaged weights:")
    for key, value in averaged_weights.items():
        print(f"{key}: {value}")

def main():
    test_serialize_deserialize()
    test_average_weights()

if __name__ == "__main__":
    main()
