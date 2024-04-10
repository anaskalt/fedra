#!/usr/bin/env python3

"""Extended test script for Peer Status, Peer Weights, and Network State classes."""

import sys
import os

# Adjust the path if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anomaly_dfl.utils.state import Status, PeerStatus, PeerWeights, NetworkState

def setup_test_peers():
    peer_status_1 = PeerStatus("peer1", Status.JOINED)
    peer_status_2 = PeerStatus("peer2", Status.TRAINING)
    peer_weights_1 = PeerWeights("peer1", {"weight": 0.5})
    peer_weights_2 = PeerWeights("peer2", {"weight": 0.7})
    return peer_status_1, peer_status_2, peer_weights_1, peer_weights_2

def display_peer_info(peer_status, peer_weights):
    print(f"Peer ID: {peer_status.peer_id}, Status: {peer_status.status.name}, Weights: {peer_weights.weights}")

def test_network_state_operations():
    print("Testing NetworkState operations...\n")
    
    # Setup test peers
    peer_status_1, peer_status_2, peer_weights_1, peer_weights_2 = setup_test_peers()
    
    # Display initial status and weights for peers
    print("Initial peer statuses and weights:")
    display_peer_info(peer_status_1, peer_weights_1)
    display_peer_info(peer_status_2, peer_weights_2)
    print()

    # Initialize NetworkState
    network_state = NetworkState()
    
    # Add peers to network state
    network_state.update_peer_status(peer_status_1)
    network_state.update_peer_status(peer_status_2)
    network_state.update_peer_weights(peer_weights_1)
    network_state.update_peer_weights(peer_weights_2)

    # Print network summary
    summary = network_state.get_network_summary()
    print("Network Summary after initial setup:")
    print(summary)
    print()

    # Update peer statuses and weights
    print("Updating peer statuses and weights...")
    print()
    peer_status_1.status = Status.READY
    peer_status_2.status = Status.ERROR
    peer_weights_1.weights = {"weight": 0.8}
    peer_weights_2.weights = {"weight": 0.2}

    # Display  status and weights for peers after update their values
    print("Updated peer statuses and weights:")
    display_peer_info(peer_status_1, peer_weights_1)
    display_peer_info(peer_status_2, peer_weights_2)
    print()

    # Reflect these updates in the network state
    network_state.update_peer_status(peer_status_1)
    network_state.update_peer_status(peer_status_2)
    network_state.update_peer_weights(peer_weights_1)
    network_state.update_peer_weights(peer_weights_2)
    
    # Print updated network summary
    updated_summary = network_state.get_network_summary()
    print("Network Summary after updates:")
    print(updated_summary)
    print()

    # Reset network state and print summary
    print("Resetting network state...")
    network_state.reset_network_state()
    reset_summary = network_state.get_network_summary()
    print("Network Summary after reset:")
    print(reset_summary)
    print()

def main():
    test_network_state_operations()

if __name__ == "__main__":
    main()
