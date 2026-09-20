"""Laser fire controller (SIMULATED, sim-only 0€, no cert/SIL claim)."""
from jetson.laser.env_interlock import env_clear, material_allows, th_inhibit
from jetson.laser.fire_controller import DutyMeter, FireController, pulse_energy_j
from jetson.laser.fire_pipeline import FirePipeline
from jetson.laser.flags import laser_enabled

__all__ = ["DutyMeter", "FireController", "FirePipeline", "env_clear",
           "laser_enabled", "material_allows", "pulse_energy_j", "th_inhibit"]
