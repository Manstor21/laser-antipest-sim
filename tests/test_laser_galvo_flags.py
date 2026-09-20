"""RED 4.1 — flags laser.enabled=false default + SIMULATED labels (PR3 sim-only)."""
import pytest
import yaml

from jetson.laser.flags import (
    CLASS_LABEL,
    LASER_ENABLED_DEFAULT,
    SIMULATED,
    SIMULATED_LABEL,
    laser_enabled,
)


def test_defaults_simulated_no_cert():
    assert LASER_ENABLED_DEFAULT is False
    assert SIMULATED is True
    assert SIMULATED_LABEL == "SIMULATED"
    assert "SIMULATED" in CLASS_LABEL
    assert "no cert" in CLASS_LABEL.lower() or "no sil" in CLASS_LABEL.lower()


def test_laser_enabled_defaults_false():
    assert laser_enabled(None) is False
    assert laser_enabled({}) is False
    assert laser_enabled({"laser": {}}) is False
    assert laser_enabled({"laser": {"enabled": False}}) is False
    assert laser_enabled({"laser": {"enabled": True}}) is True


def test_config_yaml_laser_disabled_simulated():
    with open("jetson/fusion/config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["laser"]["enabled"] is False
    assert cfg["laser"]["simulated"] is True
    assert laser_enabled(cfg) is False
