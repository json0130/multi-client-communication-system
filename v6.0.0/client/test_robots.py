"""
test_robots.py — run.py finds this lab's robots, and their configs are complete.
(The contract every new robot inherits is in test_template.py.)

Run from v6.0.0/client:  python3 -m pytest test_robots.py
"""

import glob
import importlib.util
import json

import pytest

from client import BasicClient
from run import load


def test_irobi_runs_the_generic_robot():
    from robots.generic import Robot
    assert load("irobi") is Robot


@pytest.mark.skipif(importlib.util.find_spec("navel") is not None, reason="the Navel SDK is installed here")
def test_a_robot_whose_sdk_is_missing_fails_rather_than_falling_back():
    with pytest.raises(ModuleNotFoundError) as e:
        load("navel")
    assert e.value.name == "navel"


@pytest.mark.parametrize("name", ["generic", "chatbox", "pepper", "_template"])
def test_every_robot_is_a_basic_client(name):
    assert issubclass(load(name), BasicClient)


def test_every_config_names_its_robot():
    paths = glob.glob("configs/*.json") + glob.glob("configs/*/*.json")
    assert len(paths) >= 13
    for path in paths:
        cfg = json.load(open(path))
        assert cfg["client_id"] and cfg["server_url"], path
