# p2p/peer_status_manager.py

class PeerStatus:
    def __init__(self, peerid):
        self._peerid = peerid
        self._status = "active"  # Possible values: "active", "inactive"
        self._weights = None     # To store deserialized model weights

    @property
    def peerid(self):
        return self._peerid

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, new_status):
        self._status = new_status

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        self._weights = new_weights