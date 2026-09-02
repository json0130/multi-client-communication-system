from collections import deque


class AffectStream:
    """Converts raw valence/arousal signals into smoothed PAD offsets.

    Designed to sit between a V-A emotion model and the PADEngine.
    The gain parameters scale raw model output (assumed [-1, 1]) into
    PAD-space offsets before temporal smoothing is applied.

    # TODO: plug in pretrained V-A model checkpoint here — replace
    #       mock_update() calls with real inference output from the
    #       checkpoint at Modules/models/efficientnet_HQRAF_improved_withCon.pth
    #       or a dedicated VA regression head trained on AffectNet/RAF-DB.
    """

    def __init__(
        self,
        gain_valence: float = 0.3,
        gain_arousal: float = 0.3,
        smoothing_window: int = 5,
    ):
        self.gain_valence = gain_valence
        self.gain_arousal = gain_arousal
        self._p_buf: deque[float] = deque(maxlen=smoothing_window)
        self._a_buf: deque[float] = deque(maxlen=smoothing_window)

    def update(self, valence: float, arousal: float) -> tuple[float, float]:
        """Apply temporal smoothing and return (dP, dA) offsets."""
        self._p_buf.append(valence * self.gain_valence)
        self._a_buf.append(arousal * self.gain_arousal)
        dP = sum(self._p_buf) / len(self._p_buf)
        dA = sum(self._a_buf) / len(self._a_buf)
        return dP, dA

    def mock_update(self, valence: float = 0.0, arousal: float = 0.0) -> tuple[float, float]:
        """For testing without a live camera feed."""
        return self.update(valence, arousal)
