"""DummyForcing: classify heads into first, middle, and last-frame banks."""
from .spec import MethodSpec

SPEC = MethodSpec(
    "dummy_forcing",
    "DummyForcing",
    "dummyforcing_reindex",
    "dummy_forcing",
    {
        "first_history_frames": 4,
        "middle_history_frames": 8,
        "last_history_frames": 4,
        "middle_bank_frames": 4,
        "middle_recent_frames": 4,
    },
)

