from enum import Enum, auto

class Status(Enum):
    """
    Defines possible statuses for peers within a Decentralized Federated Learning (DFL) network.
    
    Attributes:
        JOINED: Indicates a peer has successfully joined the network. This status could be used 
                to signify that initial handshakes or authentication processes have been completed.
        TRAINING: Indicates that a peer is currently engaging in the local training of its model. 
                  This could encompass data preparation, model initialization, and the execution 
                  of training epochs.
        READY: Suggests that a peer has completed its current training cycle and is ready to participate 
               in federated averaging or to undertake another round of training, depending on the 
               network's protocol.
        COMPLETED: Signifies that a peer has completed its participation in the current global model 
                   training cycle, including any post-averaging validation steps required by the network.
        ERROR: Denotes that an error has occurred in a peer's training process or network communication, 
               requiring attention or intervention.
        EXITED: Indicates that a peer has left the network or has been decommissioned, possibly after 
                successfully completing its contribution to the model or due to errors or network policies.
        NONE: An initial or reset state, indicating that the peer's current status is not set or has been 
              cleared. This can be used to represent a newly instantiated peer object awaiting to join 
              the network or a peer that has been reset for a new training cycle.
              
    These states are designed to provide a comprehensive overview of a peer's lifecycle and activities 
    within a DFL network, facilitating management, debugging, and optimization of the learning process.
    """
    JOINED = auto()
    TRAINING = auto()
    READY = auto()
    COMPLETED = auto()
    ERROR = auto()
    EXITED = auto()
    NONE = auto()


class PeerStatus:
    """Represents a single peer's status."""
    def __init__(self, peer_id: str, status: Status):
        self._peer_id = peer_id
        self._status = status

    @property
    def peer_id(self) -> str:
        """Gets the peer's ID."""
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value: str):
        """Sets the peer's ID."""
        self._peer_id = value

    @property
    def status(self) -> Status:
        """Gets the peer's status."""
        return self._status

    @status.setter
    def status(self, value: Status):
        """Sets the peer's status."""
        self._status = value

class PeerWeights:
    """Represents a single peer's weights."""
    def __init__(self, peer_id: str, weights: dict):
        self._peer_id = peer_id
        self._weights = weights

    @property
    def peer_id(self) -> str:
        """Gets the peer's ID."""
        return self._peer_id

    @peer_id.setter
    def peer_id(self, value: str):
        """Sets the peer's ID."""
        self._peer_id = value

    @property
    def weights(self) -> dict:
        """Gets the peer's weights."""
        return self._weights

    @weights.setter
    def weights(self, value: dict):
        """Sets the peer's weights."""
        self._weights = value


class NetworkState:
    """Encapsulates the network state, managing statuses and weights of all peers."""
    def __init__(self):
        self.peer_statuses = {}
        self.peer_weights = {}

    def update_peer_status(self, peer_status_info: PeerStatus):
        """Updates or adds the status of a peer."""
        self.peer_statuses[peer_status_info.peer_id] = peer_status_info.status

    def update_peer_weights(self, peer_weight_info: PeerWeights):
        """Updates or adds the weights of a peer."""
        self.peer_weights[peer_weight_info.peer_id] = peer_weight_info.weights

    def remove_peer(self, peer_id: str):
        """Removes a peer from the network state."""
        self.peer_statuses.pop(peer_id, None)
        self.peer_weights.pop(peer_id, None)

    def reset_network_state(self):
        """Resets the network state to its initial condition."""
        self.peer_statuses.clear()
        self.peer_weights.clear()

    '''def check_state(self, state: Status) -> bool:
        """Checks if all peers are in the specified state."""
        return all(status == state for status in self.peer_statuses.values())'''

    def check_state(self, *states) -> bool:
        """Checks if all peers are in any of the specified states.
        
        Args:
            \*states (Status): Variable number of state arguments to check against.
        
        Returns:
            bool: True if all peers are in any of the specified states, False otherwise.
        """
        return all(status in states for status in self.peer_statuses.values())

    def get_all_weights(self) -> dict:
        """Returns the weights of all peers."""
        return self.peer_weights.copy()

    def get_network_summary(self) -> dict:
        """Returns a summary of the network state."""
        statuses_summary = {peer_id: status.name for peer_id, status in self.peer_statuses.items()}
        return {"statuses": statuses_summary, "weights": self.get_all_weights()}