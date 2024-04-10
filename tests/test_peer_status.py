import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from anomaly_dfl.network.peer_status_manager import PeerStatusManager, PeerStatus

async def test_peer_status_manager():
    manager = PeerStatusManager()
    peer_id = "peer1"
    local_model = "local_data_local_data_local_data_local_data_local_data_local_data_local_data_local_data_local_data_local_data_local_data_"
    global_model = "global_data_global_data_global_data_global_data_global_data_global_data_global_data_global_data_global_data_global_data_"

    # Add a new peer and update its status
    await manager.update_peer_status(peer_id, PeerStatus.TRAINING)
    # Update peer models
    await manager.update_peer_models(peer_id, local_model=local_model, global_model=global_model)
    # Retrieve peer info and check values
    peer_info = await manager.get_peer_info(peer_id)

    print(peer_info.peer_id)
    print(peer_info.status)  # Now this prints the status name, not the Enum member
    print(peer_info.local_model)
    print(peer_info.global_model)

    assert peer_info.peer_id == peer_id
    # Compare against the name of the Enum member
    assert peer_info.status == PeerStatus.TRAINING.name
    assert peer_info.local_model == local_model
    assert peer_info.global_model == global_model

    # Remove peer and check if removed
    await manager.remove_peer(peer_id)
    assert await manager.get_peer_info(peer_id) is None

    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(test_peer_status_manager())
