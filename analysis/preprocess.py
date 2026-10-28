"""PRACH record parsing, timing recovery, CFO correction and normalisation.

A record is int16 little-endian interleaved I/Q:
    4 * (capture_samples + 139) bytes
    capture_samples = 9000 (UE captures) or 9600 (model captures)
    the last 139 complex values are the frequency-domain PRACH bins.

The 576 + 4*2048 layout is the long NR PRACH preamble; timing is recovered from
the CP/tail coherence peak rather than a fixed offset, because the capture is not
trigger-aligned.
"""

import numpy as np

from . import config


def load_capture(path):
    """Return (time_iq, frequency_bins, capture_samples)."""
    raw = path.read_bytes()
    if len(raw) % 4:
        raise ValueError(f"{path}: length {len(raw)} is not a whole number of int16 I/Q pairs")
    interleaved = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float64)
    z = interleaved[:, 0] + 1j * interleaved[:, 1]
    capture = len(z) - config.FREQ_BINS
    if capture not in config.CAPTURE_SAMPLES:
        raise ValueError(f"{path}: unsupported capture length {capture}")
    return z[:capture], z[capture:], capture


def _window_sums(values, length):
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return cumulative[length:] - cumulative[:-length]


def cp_profile(x):
    """Normalised squared coherence between the CP and the tail one body away."""
    lag = config.BODY
    head, tail = x[:-lag], x[lag:]
    cross = _window_sums(np.conj(head) * tail, config.CP)
    e_head = _window_sums(np.abs(head) ** 2, config.CP)
    e_tail = _window_sums(np.abs(tail) ** 2, config.CP)
    return np.clip(np.abs(cross) ** 2 / (e_head * e_tail + 1e-12), 0.0, 1.0)


def analyze(x):
    """Timing, CFO and quality metrics for one capture window."""
    profile = cp_profile(x)
    start = int(np.argmax(profile))
    if start + config.CP + config.BODY > len(x):
        raise ValueError("Timing peak leaves no room for a full PRACH body")
    body = x[start + config.CP:start + config.CP + config.BODY]
    repeats = body.reshape(config.REPETITIONS, config.PERIOD)
    cross = np.sum(np.conj(repeats[:-1]) * repeats[1:], axis=1)
    energy = np.sum(np.abs(repeats) ** 2, axis=1)
    coherence = np.clip(np.abs(cross) ** 2 / (energy[:-1] * energy[1:] + 1e-12), 0.0, 1.0)
    radians = float(np.angle(np.sum(cross)) / config.PERIOD)
    return {
        "sync_start": start,
        "cp_corr": float(profile[start]),
        "min_repeat_corr": float(coherence.min()),
        "cfo_rad_per_sample": radians,
        "cfo_hz": radians * config.SAMPLE_RATE / (2 * np.pi),
        "origin": int(x.__array_interface__["data"][0]),
    }


def transform(x, metrics, compensate_cfo=True):
    """Crop the PRACH body, optionally undo the repeat-derived CFO, RMS-scale."""
    start = metrics["sync_start"] + config.CP
    body = x[start:start + config.BODY].astype(np.complex128, copy=True)
    if body.size != config.BODY:
        raise ValueError("Capture window does not contain a full PRACH body")
    if compensate_cfo:
        n = np.arange(config.BODY)
        body *= np.exp(-1j * metrics["cfo_rad_per_sample"] * n)
    rms = float(np.sqrt(np.mean(np.abs(body) ** 2)))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Degenerate body energy")
    body /= rms
    return np.stack((body.real, body.imag)).astype(np.float32)


def prepare(path, compensate_cfo=True, metrics=None):
    x, _frequency, _capture = load_capture(path)
    if metrics is None:
        metrics = analyze(x)
    return transform(x, metrics, compensate_cfo), metrics
