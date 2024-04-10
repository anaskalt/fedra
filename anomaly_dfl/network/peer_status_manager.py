"""
Manages the statuses and models of peers in a Federated Learning P2P network.

This module provides classes to represent the status of peers in the network,
manage their information (including local and global models), and facilitate
synchronized status updates across peers in a thread-safe manner.

Classes:
    PeerStatus: Enumerates possible states of peers within the network.
    PeerInfo: Represents a peer's information, including status and models.
    PeerStatusManager: Manages and updates the status and models of all peers.

Example:
    peer_status_manager = PeerStatusManager()
    asyncio.run(peer_status_manager.update_peer_status('peer1', PeerStatus.TRAINING))
    asyncio.run(peer_status_manager.update_peer_models('peer1', local_model=model.state_dict()))
"""

import asyncio
from enum import Enum, auto

class PeerStatus(Enum):
    """
    Enumerates possible states of peers in a Federated Learning P2P network.

    Attributes:
        JOINED: Peer has joined the network.
        TRAINING: Peer is currently training its local model.
        READY: Peer is ready for federated averaging.
        COMPLETED: Peer has completed its training and federated averaging.
        ERROR: An error has occurred in the peer's training or communication.
        EXITED: Peer has exited the network.
        NONE: Initial or reset state, indicating no current status.
    """
    JOINED = auto()
    TRAINING = auto()
    READY = auto()
    COMPLETED = auto()
    ERROR = auto()
    EXITED = auto()
    NONE = auto()

class PeerInfo:
    """
    Represents information related to a peer, including its status and models.

    Attributes:
        peer_id (str): The unique identifier of the peer.
        status (PeerStatus): The current status of the peer.
        local_model (dict): The peer's local model weights.
        global_model (dict): The global model weights received from federated averaging.
    """

    def __init__(self, peer_id, status=PeerStatus.NONE, local_model=None, global_model=None):
        self._peer_id = peer_id
        self._status = status
        self._local_model = local_model
        self._global_model = global_model

    @property
    def peer_id(self):
        """Returns the unique identifier of the peer."""
        return self._peer_id

    @property
    def status(self):
        """Returns the name of the current status of the peer."""
        return self._status.name

    @status.setter
    def status(self, new_status):
        """Sets a new status for the peer."""
        self._status = new_status

    @property
    def local_model(self):
        """Returns the local model weights of the peer."""
        return self._local_model

    @local_model.setter
    def local_model(self, new_local_model):
        """Updates the local model weights of the peer."""
        self._local_model = new_local_model

    @property
    def global_model(self):
        """Returns the global model weights received by the peer."""
        return self._global_model

    @global_model.setter
    def global_model(self, new_global_model):
        """Updates the global model weights for the peer."""
        self._global_model = new_global_model

class PeerStatusManager:
    """
    Manages the statuses and models of peers in the network in a thread-safe manner.

    Provides methods to update peer statuses, models, retrieve peer information,
    remove peers from management, and reset the statuses and models of all peers.
    """

    def __init__(self):
        self.peers = {}
        self.peers_lock = asyncio.Lock()

    async def update_peer_status(self, peer_id, new_status):
        """
        Asynchronously updates the status of a given peer.
        
        Args:
            peer_id (str): The unique identifier of the peer.
            new_status (PeerStatus): The new status to be assigned to the peer.
        """
        async with self.peers_lock:
            if peer_id not in self.peers:
                self.peers[peer_id] = PeerInfo(peer_id)
            self.peers[peer_id].status = new_status

    async def update_peer_models(self, peer_id, local_model=None, global_model=None):
        """
        Asynchronously updates the local and/or global models of a given peer.
        
        Args:
            peer_id (str): The unique identifier of the peer.
            local_model (dict): The new local model weights, if any.
            global_model (dict): The new global model weights, if any.
        """
        async with self.peers_lock:
            if peer_id in self.peers:
                if local_model is not None:
                    self.peers[peer_id].local_model = local_model
                if global_model is not None:
                    self.peers[peer_id].global_model = global_model

    async def get_peer_info(self, peer_id):
        """
        Asynchronously retrieves information about a given peer.

        Args:
            peer_id (str): The unique identifier of the peer.

        Returns:
            PeerInfo: The information of the requested peer, if found.
        """
        async with self.peers_lock:
            return self.peers.get(peer_id, None)

    async def remove_peer(self, peer_id):
        """
        Asynchronously removes a peer from management.
        
        Args:
            peer_id (str): The unique identifier of the peer to be removed.
        """
        async with self.peers_lock:
            if peer_id in self.peers:
                del self.peers[peer_id]

    async def reset_all_peers(self):
        """
        Asynchronously resets the status, local model, and global model for all peers.
        """
        async with self.peers_lock:
            for peer in self.peers.values():
                peer.status = PeerStatus.NONE
                peer.local_model = None
                peer.global_model = None
