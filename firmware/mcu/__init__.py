"""MCU package (SIMULATED, sim-only 0EUR, no HW)."""
from firmware.mcu.virtual_mcu import ORDER, TICK_S, Task, VirtualMcu
from firmware.mcu.watchdog import TIMEOUT_S, McuWatchdog

__all__ = ["ORDER", "TICK_S", "Task", "VirtualMcu", "McuWatchdog", "TIMEOUT_S"]
