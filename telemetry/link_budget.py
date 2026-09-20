"""Link budget sim sobre LinkSim (SIMULATED, sim-only 0EUR).

Design #171: `telemetry/link_budget.py` fórmulas puras sobre
`LinkSim.dropped/queue+delay_for()`; LoRa SF12/BW125 y WiFi MCS
parametrizados. Solo lectura: nunca muta `sim/link/*` (B1-4);
test verifica hash pre/post igual.

Fórmula Friis free-space (documentada, determinista):
  FSPL(dB) = 20*log10(d_km) + 20*log10(f_MHz) + 32.44
  Lmax(dB) = tx_dbm - sens_dbm
  range_km = 10**((Lmax - 20*log10(f_MHz) - 32.44) / 20)
Congestión (LinkSim FIFO16): goodput = 1/(1+dropped+depth/capacity);
  effective_range_km = range_km * goodput; warning si dropped>0 o
  cola llena. SNR de referencia a 1km: rssi_1km - sens_dbm.
Sim-time `t` explícito float. Contratos:
  compute(link, radio, tx_dbm, sens_dbm, t=0.0, nbytes=32) -> dict
"""

import math

SIMULATED = True

LORA_FREQ_MHZ = 868.0
WIFI_FREQ_MHZ = 2437.0

LORA_RADIOS = ("lora_sf12_bw125", "lora_sf12", "lora")
WIFI_MCS_SENS_DEFAULT = {
    "wifi_mcs0": -82.0,
    "wifi_mcs1": -79.0,
    "wifi_mcs2": -77.0,
    "wifi_mcs3": -74.0,
    "wifi_mcs4": -70.0,
    "wifi_mcs5": -66.0,
    "wifi_mcs6": -65.0,
    "wifi_mcs7": -64.0,
}


def _freq_for(radio: str) -> float:
    r = str(radio).lower()
    if r in LORA_RADIOS:
        return LORA_FREQ_MHZ
    if r.startswith("wifi"):
        return WIFI_FREQ_MHZ
    raise ValueError(f"unknown radio {radio!r} (lora_sf12_bw125|wifi_mcsN)")


def _fspl_db(d_km: float, f_mhz: float) -> float:
    return 20.0 * math.log10(max(float(d_km), 1e-9)) + \
        20.0 * math.log10(float(f_mhz)) + 32.44


def compute(link, radio: str, tx_dbm: float, sens_dbm: float,
            t: float = 0.0, nbytes: int = 32) -> dict:
    """Calcula budget puro sobre stats de LinkSim (solo lectura)."""
    t = float(t)
    tx = float(tx_dbm)
    sens = float(sens_dbm)
    freq = _freq_for(radio)
    lmax = tx - sens
    if lmax <= 0:
        raise ValueError("tx_dbm must exceed sens_dbm")
    rng = 10.0 ** ((lmax - 20.0 * math.log10(freq) - 32.44) / 20.0)
    # Solo lectura de LinkSim: depth/dropped/delay_for (nunca send/recv).
    depth = len(list(getattr(link, "_queue", [])))
    capacity = int(getattr(link, "capacity", 16) or 16)
    dropped = int(getattr(link, "dropped", 0) or 0)
    try:
        delay_s = float(link.delay_for(int(nbytes)))
    except Exception:
        delay_s = 0.0
    goodput = 1.0 / (1.0 + float(dropped) + float(depth) / float(capacity))
    goodput = max(0.0, min(1.0, goodput))
    effective = float(rng) * float(goodput)
    rssi_1km = tx - _fspl_db(1.0, freq)
    snr = float(rssi_1km) - float(sens)
    rssi_eff = tx - _fspl_db(max(effective, 1e-9), freq)
    warning = bool(dropped > 0 or depth >= capacity)
    return {
        "radio": str(radio),
        "t": t,
        "freq_mhz": freq,
        "range_km": float(rng),
        "effective_range_km": float(effective),
        "snr_db": float(snr),
        "rssi_dbm": float(rssi_eff),
        "rssi_1km_dbm": float(rssi_1km),
        "goodput": float(goodput),
        "warning": warning,
        "delay_s": float(delay_s),
        "dropped": int(dropped),
        "depth": int(depth),
    }
