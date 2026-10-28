"""Training loop and per-run evaluation artifacts."""

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from . import config, dataset, losses, metrics, models, scorers


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 5e-4
    embedding_dim: int = 128
    head: str = "linear"
    ce_weight: float = 1.0
    maxmin_weight: float = 0.0
    supcon_weight: float = 0.0
    label_smoothing: float = 0.05
    num_classes: int = config.NUM_KNOWN
    amp: bool = True
    patience: int = 10
    acceptance: float = 0.95
    augment: bool = True
    augment_noise: float = 0.02
    augment_shift: int = 16
    scorer_names: tuple = ("msp", "energy", "prototype", "mahalanobis", "openmax")
    tag: str = "run"


def augment(batch, generator, noise_sigma=0.02, shift=16):
    """Label-preserving RF augmentations applied on the GPU per batch.

    Circular shift mimics residual timing offset, the rotation mimics residual CFO,
    gain mimics AGC drift and the noise term mimics receiver noise. All four leave
    the transmitter fingerprint intact, unlike e.g. spectral inversion.

    `noise_sigma=0.02` is about 34 dB SNR against the unit-power normalised signal;
    larger values emulate the weaker 4 m condition.
    """
    batch_size = batch.shape[0]
    shifts = torch.randint(-shift, shift + 1, (batch_size,), generator=generator,
                           device=batch.device)
    shifted = torch.stack([torch.roll(batch[i], int(shifts[i]), dims=-1)
                           for i in range(batch_size)])
    theta = (torch.rand(batch_size, generator=generator, device=batch.device) - 0.5) * 0.6
    cosine, sine = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    in_phase = shifted[:, 0] * cosine - shifted[:, 1] * sine
    quadrature = shifted[:, 0] * sine + shifted[:, 1] * cosine
    out = torch.stack([in_phase, quadrature], dim=1)
    gain = 1.0 + (torch.rand(batch_size, generator=generator, device=batch.device) - 0.5) * 0.2
    out = out * gain[:, None, None]
    return out + torch.randn_like(out) * noise_sigma


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_arrays(unknown_device=config.UNKNOWN_NAME):
    """Assemble split arrays with one or more devices held out of every known split.

    `unknown_device` may be a single name or a list, which supports the 5-known /
    3-unknown protocol. Class ids are re-mapped to 0..K-1 over the remaining known
    devices, so the same loader serves every protocol variant.
    """
    unknown_devices = ([unknown_device] if isinstance(unknown_device, str)
                       else list(unknown_device))
    cache, meta = dataset.load()
    columns = dataset.index(meta)
    is_unknown = np.isin(columns["device"], unknown_devices)
    known_devices = [d for d in config.CLASS_NAMES if d not in unknown_devices]
    remap = {device: index for index, device in enumerate(known_devices)}

    out = {}
    for split in ("train", "val", "test"):
        rows = np.flatnonzero((columns["split"] == split) & ~is_unknown)
        out[split] = {
            "X": np.asarray(cache[rows], dtype=np.float32),
            "y": np.array([remap[device] for device in columns["device"][rows]], dtype=np.int64),
            "device": columns["device"][rows],
            "root": columns["root_u"][rows],
            "source": columns["source"][rows],
        }
    rows = np.flatnonzero(is_unknown)
    out["unknown_test"] = {
        "X": np.asarray(cache[rows], dtype=np.float32),
        "y": np.full(len(rows), -1, dtype=np.int64),
        "device": columns["device"][rows],
        "root": columns["root_u"][rows],
        "source": columns["source"][rows],
    }
    return out, known_devices


@torch.no_grad()
def infer(model, X, batch_size=256, device="cuda"):
    model.eval()
    logits_out, embedding_out = [], []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start:start + batch_size]).to(device)
        logits, embedding = model(batch)
        logits_out.append(logits.float().cpu().numpy())
        embedding_out.append(embedding.float().cpu().numpy())
    return np.concatenate(logits_out), np.concatenate(embedding_out)


def train_model(cfg: TrainConfig, data=None, verbose=True):
    set_seed(cfg.seed)
    if data is None:
        data, _known = load_arrays()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train = data["train"]
    val = data["val"]
    model = models.build(cfg.embedding_dim, cfg.num_classes, cfg.head).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train["X"]) // cfg.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=cfg.epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device == "cuda")

    generator = torch.Generator().manual_seed(cfg.seed)
    augment_generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train["X"]), torch.from_numpy(train["y"])),
        batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=generator,
        num_workers=0, pin_memory=device == "cuda",
    )

    best = {"accuracy": -1.0, "epoch": -1, "state": None}
    history = []
    stale = 0
    started = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        totals, counts = {}, 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            if cfg.augment:
                batch_x = augment(batch_x, augment_generator,
                                  noise_sigma=cfg.augment_noise,
                                  shift=cfg.augment_shift)
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device == "cuda"):
                logits, embedding = model(batch_x, batch_y if cfg.head == "cosface" else None)
            loss, terms = losses.combined(
                logits.float(), embedding.float(), batch_y, cfg.num_classes,
                ce_weight=cfg.ce_weight, maxmin_weight=cfg.maxmin_weight,
                supcon_weight=cfg.supcon_weight, label_smoothing=cfg.label_smoothing)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            for key, value in terms.items():
                totals[key] = totals.get(key, 0.0) + value
            counts += 1

        logits, _ = infer(model, val["X"], device=device)
        accuracy = float((logits.argmax(1) == val["y"]).mean())
        record = {"epoch": epoch, "val_accuracy": accuracy,
                  **{k: v / max(counts, 1) for k, v in totals.items()}}
        history.append(record)
        if verbose:
            print(f"  epoch {epoch:3d} loss {record['loss']:.4f} val_acc {accuracy:.4f}", flush=True)

        if accuracy > best["accuracy"] + 1e-6:
            best = {"accuracy": accuracy, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            if verbose:
                print(f"  early stop at epoch {epoch}", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, {"history": history, "best_val_accuracy": best["accuracy"],
                   "best_epoch": best["epoch"],
                   "train_seconds": round(time.time() - started, 1),
                   "device": device}


def evaluate_run(model, cfg: TrainConfig, data, training):
    """Fit every scorer on known train statistics, then evaluate blind."""
    logits_train, embedding_train = infer(model, data["train"]["X"])
    fitted = scorers.fit_all(logits_train, embedding_train, data["train"]["y"],
                             cfg.scorer_names)

    parts = {}
    for split in ("val", "test", "unknown_test"):
        parts[split] = infer(model, data[split]["X"])

    results = {}
    unknown_devices = data["unknown_test"]["device"]
    for name, scorer in fitted.items():
        logits_val, embedding_val = parts["val"]
        logits_test, embedding_test = parts["test"]
        logits_unknown, embedding_unknown = parts["unknown_test"]

        knownness_val, _ = scorer.score(logits_val, embedding_val)
        threshold = metrics.threshold_at_acceptance(knownness_val, cfg.acceptance)
        knownness_test, prediction_test = scorer.score(logits_test, embedding_test)
        knownness_unknown, prediction_unknown = scorer.score(logits_unknown, embedding_unknown)

        results[name] = metrics.evaluate(
            knownness_test, prediction_test, data["test"]["y"],
            knownness_unknown, prediction_unknown, threshold)
        results[name]["validation_acceptance"] = float((knownness_val >= threshold).mean())
        results[name]["unknown_by_device"] = {
            str(device): float((knownness_unknown[unknown_devices == device] < threshold).mean())
            for device in np.unique(unknown_devices)
        }

    return results, {"train": (logits_train, embedding_train)}


def run(cfg: TrainConfig, output_dir=None, save=True, unknown_device=config.UNKNOWN_NAME,
        data=None, verbose=True):
    unknown_devices = ([unknown_device] if isinstance(unknown_device, str)
                       else list(unknown_device))
    if data is None:
        data, known_devices = load_arrays(unknown_devices)
    else:
        known_devices = [d for d in config.CLASS_NAMES if d not in unknown_devices]
    cfg.num_classes = len(known_devices)
    model, training = train_model(cfg, data, verbose=verbose)
    results, _ = evaluate_run(model, cfg, data, training)

    summary = {
        "config": asdict(cfg),
        "unknown_device": unknown_devices[0] if len(unknown_devices) == 1 else unknown_devices,
        "unknown_devices": unknown_devices,
        "known_devices": known_devices,
        "training": {k: v for k, v in training.items() if k != "history"},
        "results": results,
    }
    if save and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(
            json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        (output_dir / "history.json").write_text(
            json.dumps(training["history"], indent=2), encoding="utf-8")
        (output_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), output_dir / "model.pt")
    return summary, model, data
