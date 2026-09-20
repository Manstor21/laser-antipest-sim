"""RED 3.2 — T/H inhibit + material map (PR3 sim-only, 0EUR)."""
import pytest

from jetson.laser.env_interlock import (
    MATERIAL_MAP,
    env_clear,
    material_allows,
    th_inhibit,
)


def test_th_nominal_clears():
    assert th_inhibit(22.0, 55.0) is False
    assert th_inhibit(30.0, 30.0) is False
    assert env_clear(22.0, 55.0, "soil") is True


def test_th_hot_dry_inhibits():
    assert th_inhibit(36.0, 55.0) is True
    assert th_inhibit(22.0, 15.0) is True
    assert th_inhibit(32.0, 25.0) is True
    assert env_clear(36.0, 55.0, "soil") is False
    assert env_clear(22.0, 55.0, "soil") is True


def test_th_missing_fails_safe():
    assert th_inhibit(None, 55.0) is True
    assert th_inhibit(22.0, None) is True
    assert env_clear(None, None, "soil") is False


def test_material_map_known_vs_unknown():
    assert "dry_grass" in MATERIAL_MAP
    assert material_allows("dry_grass") is True
    assert material_allows("soil") is True
    assert material_allows("unknown") is False
    assert material_allows(None) is False
    assert env_clear(22.0, 55.0, "unknown") is False
