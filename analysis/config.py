"""Central constants for the PRACH open-set RFF experiments.

Everything resolves relative to the project root; nothing is written outside it.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
CACHE_DIR = DATA / "cache"
CACHE_NPY = CACHE_DIR / "cache_cfo.npy"
CACHE_META = CACHE_DIR / "meta.json"
SPLIT_DIR = DATA / "splits" / "open_set_2m_balanced_seed42"
OUTPUT_DIR = ROOT / "outputs"

# Known devices in class_id order; UE20 is the held-out unknown device.
CLASS_NAMES = ["UE9", "UE12", "UE17", "model1", "model2", "model5", "model9"]
UNKNOWN_NAME = "UE20"
NUM_KNOWN = len(CLASS_NAMES)
UNKNOWN_LABEL = -1

# 2 m pool. Devices without a distance sub-directory are flat captures recorded at 2 m.
DEVICE_DIRS = {
    "UE9": "data/UE9/UE9_honor_2m",
    "UE12": "data/UE12/UE12_huawei_2m",
    "UE17": "data/UE17/UE17_xiaomi_2m",
    "model1": "data/model1",
    "model2": "data/model2",
    "model5": "data/model5",
    "model9": "data/model9",
    "UE20": "data/UE20",
}

# PRACH geometry: long preamble, CP 576 + 4 x 2048 body, plus 139 DFT bins.
CP = 576
PERIOD = 2048
REPETITIONS = 4
BODY = PERIOD * REPETITIONS
PREAMBLE = CP + BODY            # 8768 samples from CP start to end of the 4th block
FREQ_BINS = 139
SAMPLE_RATE = 61_440_000.0
# Time-domain lengths observed in the two capture families (UE vs. model rig).
CAPTURE_SAMPLES = (9000, 9600)

# Split policy.
SEED = 42
PER_CLASS_KNOWN = 800
KNOWN_FRACTIONS = (0.60, 0.20, 0.20)   # train / val / test
UNKNOWN_TEST = 800
ROOT_MODULUS = 139                      # ZC sequence length used for the root estimate

SPLITS = ("train", "val", "test", "unknown_test")


def split_path(name):
    return SPLIT_DIR / f"{name}.jsonl"
