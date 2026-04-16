import pytest

from src.train import _enforce_pre_event_policy


def test_pre_event_policy_rejects_post_event_features():
    with pytest.raises(RuntimeError):
        _enforce_pre_event_policy("pre_event", ["number_of_vehicles", "number_of_casualties"])


def test_post_event_policy_allows_post_event_features():
    _enforce_pre_event_policy("post_event", ["number_of_vehicles", "number_of_casualties"])
