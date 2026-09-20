"""RED 1.2 — McuWatchdog 10ms independiente (must fail before impl, PR1)."""
from firmware.mcu.watchdog import McuWatchdog


def test_watchdog_expires_after_10ms():
    wdog = McuWatchdog()
    wdog.feed(t=1.000)
    assert wdog.expired(t=1.005) is False
    assert wdog.expired(t=1.011) is True


def test_watchdog_feed_resets():
    wdog = McuWatchdog()
    wdog.feed(t=0.0)
    assert wdog.expired(t=0.011) is True
    wdog.feed(t=0.012)
    assert wdog.expired(t=0.015) is False
    assert wdog.expired(t=0.023) is True


def test_watchdog_boundary_exactly_10ms_not_expired():
    wdog = McuWatchdog()
    wdog.feed(t=2.0)
    # Contract: expired iff t - last > 0.010 (strict).
    assert wdog.expired(t=2.010) is False
    assert wdog.expired(t=2.010001) is True
