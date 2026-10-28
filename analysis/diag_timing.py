"""Diagnose PRACH timing recovery: is the CP peak sharp or a plateau?

If the four 2048-sample preamble blocks are identical, the CP/tail coherence is ~1
for every lag that keeps both windows inside the periodic region, so argmax is an
arbitrary point on a plateau. That would inject a random circular shift into every
training sample. This script measures the plateau and compares three anchors:
argmax, the left plateau edge, and the energy onset.
"""

import sys

import numpy as np

from . import config, dataset, preprocess


def plateau_bounds(profile, fraction=0.9):
    peak = profile.max()
    above = np.flatnonzero(profile >= fraction * peak)
    return int(above[0]), int(above[-1]), float(peak)


def energy_onset(x, threshold=0.5):
    smooth = np.convolve(np.abs(x) ** 2, np.ones(32) / 32, mode="same")
    reference = float(np.median(smooth[1000:8000]))
    above = np.flatnonzero(smooth > threshold * reference)
    return int(above[0]) if len(above) else -1


def main(count=12):
    cache, meta = dataset.load()
    columns = dataset.index(meta)
    rows = np.flatnonzero(columns["split"] == "train")[:count]

    print(f"{'device':8s} {'argmax':>6s} {'left':>5s} {'right':>5s} {'width':>6s} "
          f"{'onset':>6s} {'peak':>6s}")
    starts, lefts = [], []
    for row in rows:
        x, _freq, _cap = preprocess.load_capture(config.ROOT / columns["source"][row])
        profile = preprocess.cp_profile(x)
        left, right, peak = plateau_bounds(profile)
        onset = energy_onset(x)
        starts.append(int(np.argmax(profile)))
        lefts.append(left)
        print(f"{columns['device'][row]:8s} {starts[-1]:6d} {left:5d} {right:5d} "
              f"{right - left:6d} {onset:6d} {peak:6.4f}")

    print(f"\nargmax std = {np.std(starts):.2f}, left-edge std = {np.std(lefts):.2f}")
    print(f"argmax range = {min(starts)}..{max(starts)}, left range = {min(lefts)}..{max(lefts)}")

    # How different is the cropped waveform under the two anchors?
    row = rows[0]
    x, _f, _c = preprocess.load_capture(config.ROOT / columns["source"][row])
    profile = preprocess.cp_profile(x)
    left, _right, _peak = plateau_bounds(profile)
    for name, anchor in (("argmax", int(np.argmax(profile))), ("left", left)):
        body = x[anchor + config.CP: anchor + config.CP + config.BODY]
        repeats = body.reshape(config.REPETITIONS, config.PERIOD)
        cross = np.sum(np.conj(repeats[:-1]) * repeats[1:], axis=1)
        energy = np.sum(np.abs(repeats) ** 2, axis=1)
        coherence = np.abs(cross) ** 2 / (energy[:-1] * energy[1:] + 1e-12)
        print(f"{name:6s} anchor={anchor:4d} repeat-coherence={np.round(coherence, 5)}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
