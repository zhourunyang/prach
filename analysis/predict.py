"""Inference: raw PRACH capture -> feature vector, predicted device, confidence.

The confidence reported is the softmax maximum (MSP). Because this project is
open-set, the same call also returns a calibrated `knownness` and the resulting
accept/reject decision, so a caller can refuse to assign a class at all.

Library use
-----------
    from analysis.predict import Predictor
    predictor = Predictor("outputs/main/ce_seed42")
    out = predictor.predict("data/UE20/prach_1.bin")
    out["feature"].shape   # (128,)
    out["device"]          # 'UE9' ... predicted class name
    out["confidence"]      # 0.0 - 1.0

Command line
------------
    python -m analysis.predict data/UE20/*.bin --run outputs/main/ce_seed42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import config, models, preprocess, scorers, train


class Predictor:
    """Loads one trained run and exposes single-capture / batch inference."""

    def __init__(self, run_dir=None, acceptance=0.95, device=None):
        self.run_dir = Path(run_dir or (Path(config.OUTPUT_DIR) / "main" / "ce_seed42"))
        if not (self.run_dir / "model.pt").exists():
            raise FileNotFoundError(f"no model.pt under {self.run_dir}")
        self.cfg = json.loads((self.run_dir / "config.json").read_text(encoding="utf-8"))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.acceptance = acceptance

        self.model = models.build(self.cfg["embedding_dim"], self.cfg["num_classes"],
                                  self.cfg["head"])
        self.model.load_state_dict(
            torch.load(self.run_dir / "model.pt", map_location="cpu"))
        self.model.to(self.device).eval()

        summary = json.loads((self.run_dir / "results.json").read_text(encoding="utf-8"))
        self.known_devices = summary["known_devices"]
        self._load_openset()

    # ---- calibration -------------------------------------------------------
    def _cache_path(self):
        return self.run_dir / f"openset_acceptance{self.acceptance:.2f}.npz"

    def _load_openset(self):
        cache = self._cache_path()
        if cache.exists():
            stored = np.load(cache, allow_pickle=True)
            self.pca_mean = stored["pca_mean"]
            self.pca_components = stored["pca_components"]
            self.class_means = stored["class_means"]
            self.precisions = stored["precisions"]
            self.threshold = float(stored["threshold"])
            return

        data, _known = train.load_arrays()
        logits, embedding = train.infer(self.model, data["train"]["X"], device=self.device)
        mahalanobis = scorers.Mahalanobis().fit(logits, embedding, data["train"]["y"])
        val_logits, val_embedding = train.infer(self.model, data["val"]["X"],
                                                device=self.device)
        knownness_val, _ = mahalanobis.score(val_logits, val_embedding)
        self.pca_mean = mahalanobis.pca.mean_
        self.pca_components = mahalanobis.pca.components_
        self.class_means = np.stack(mahalanobis.means)
        self.precisions = np.stack(mahalanobis.precisions)
        self.threshold = float(np.quantile(knownness_val, 1.0 - self.acceptance,
                                           method="lower"))
        np.savez(cache, pca_mean=self.pca_mean, pca_components=self.pca_components,
                 class_means=self.class_means, precisions=self.precisions,
                 threshold=self.threshold)

    def _knownness(self, embedding):
        projected = (embedding - self.pca_mean) @ self.pca_components.T
        distances = np.empty((len(projected), len(self.class_means)))
        for index, (mean, precision) in enumerate(zip(self.class_means, self.precisions)):
            difference = projected - mean
            distances[:, index] = np.einsum("ij,jk,ik->i", difference, precision, difference)
        return -np.sqrt(np.clip(distances, 0, None)).min(1)

    # ---- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict_array(self, X, top_k=1):
        """X: float32 [N, 2, 8192] already preprocessed."""
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None]
        logits_out, embedding_out = [], []
        for start in range(0, len(X), 128):
            batch = torch.from_numpy(X[start:start + 128]).to(self.device)
            logits, embedding = self.model(batch)
            logits_out.append(logits.float().cpu().numpy())
            embedding_out.append(embedding.float().cpu().numpy())
        logits = np.concatenate(logits_out)
        embedding = np.concatenate(embedding_out)

        shifted = logits - logits.max(1, keepdims=True)
        probability = np.exp(shifted) / np.exp(shifted).sum(1, keepdims=True)
        knownness = self._knownness(embedding)

        results = []
        for index in range(len(X)):
            order = np.argsort(-probability[index])[:top_k]
            accepted = bool(knownness[index] >= self.threshold)
            results.append({
                "device": self.known_devices[int(order[0])],
                "class_id": int(order[0]),
                "confidence": float(probability[index, order[0]]),
                "feature": embedding[index].astype(np.float64),
                "knownness": float(knownness[index]),
                "is_known": accepted,
                "device_if_known": self.known_devices[int(order[0])] if accepted else "UNKNOWN",
                "top_k": [{"device": self.known_devices[int(i)],
                           "probability": float(probability[index, i])} for i in order],
            })
        return results

    def predict(self, capture_path, top_k=1):
        values, _metrics = preprocess.prepare(Path(capture_path))
        result = self.predict_array(values[None], top_k=top_k)[0]
        result["file"] = str(capture_path)
        return result

    def predict_many(self, capture_paths, top_k=1):
        paths = [Path(p) for p in capture_paths]
        X = np.stack([preprocess.prepare(path)[0] for path in paths])
        results = self.predict_array(X, top_k=top_k)
        for path, result in zip(paths, results):
            result["file"] = str(path)
        return results


def to_jsonable(record, digits=6):
    output = {}
    for key, value in record.items():
        if key == "feature":
            output[key] = [round(float(v), digits) for v in value]
        elif isinstance(value, (np.floating, float)):
            output[key] = round(float(value), digits)
        elif isinstance(value, np.integer):
            output[key] = int(value)
        else:
            output[key] = value
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", nargs="+", help=".bin files or directories")
    parser.add_argument("--run", default=str(Path(config.OUTPUT_DIR) / "main" / "ce_seed42"))
    parser.add_argument("--acceptance", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", type=Path, help="write full records (incl. 128-d feature)")
    parser.add_argument("--limit", type=int, help="only the first N captures overall")
    parser.add_argument("--summary-only", action="store_true",
                        help="print aggregate statistics instead of one row per capture")
    parser.add_argument("--per-dir", type=int,
                        help="take at most N captures from each directory argument")
    parser.add_argument("--sample", action="store_true",
                        help="with --per-dir, draw a seeded random sample instead of the first N")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    paths = []
    for item in args.captures:
        path = Path(item)
        if path.is_dir():
            found = sorted(path.glob("*.bin"))
            if args.per_dir and len(found) > args.per_dir:
                if args.sample:
                    picked = np.sort(rng.choice(len(found), args.per_dir, replace=False))
                    found = [found[i] for i in picked]
                else:
                    found = found[:args.per_dir]
            paths.extend(found)
        else:
            paths.append(path)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit("no .bin captures found")

    predictor = Predictor(args.run, acceptance=args.acceptance)
    records = predictor.predict_many(paths, top_k=args.top_k)

    if args.summary_only:
        accepted = [record for record in records if record["is_known"]]
        rejected = [record for record in records if not record["is_known"]]
        print(f"captures            {len(records)}")
        print(f"rejected as unknown {len(rejected):4d}  ({len(rejected) / len(records):.1%})")
        print(f"accepted as known   {len(accepted):4d}  ({len(accepted) / len(records):.1%})")
        if accepted:
            counts = {}
            for record in accepted:
                counts[record["device"]] = counts.get(record["device"], 0) + 1
            detail = ", ".join(f"{k} {v}" for k, v in
                               sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"  accepted as       {detail}")
        else:
            print("  accepted as       -")
        if rejected:
            counts = {}
            for record in rejected:
                counts[record["device"]] = counts.get(record["device"], 0) + 1
            detail = ", ".join(f"{k} {v}" for k, v in
                               sorted(counts.items(), key=lambda kv: -kv[1]))
            print(f"  rejected, top-1   {detail}")
        confidence = np.array([record["confidence"] for record in records])
        knownness = np.array([record["knownness"] for record in records])
        print(f"confidence          mean {confidence.mean():.3f}  "
              f"median {np.median(confidence):.3f}  "
              f"min {confidence.min():.3f}  max {confidence.max():.3f}")
        print(f"knownness           mean {knownness.mean():.3f}  "
              f"median {np.median(knownness):.3f}  "
              f"min {knownness.min():.3f}  max {knownness.max():.3f}")
        print(f"threshold tau       {predictor.threshold:.4f}  "
              f"(acceptance target {args.acceptance:.0%})")
        print(f"feature dim         {len(records[0]['feature'])}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps([to_jsonable(r) for r in records],
                                            ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {args.json}")
        return

    print(f"{'capture':44s} {'class':8s} {'conf':>6s} {'knownness':>10s}  decision")
    for record in records:
        label = f"{Path(record['file']).parent.name}/{Path(record['file']).name}"
        label = label if len(label) <= 44 else "..." + label[-41:]
        print(f"{label:44s} {record['device']:8s} {record['confidence']:6.3f} "
              f"{record['knownness']:10.3f}  {record['device_if_known']}")

    recovered = sum(record["is_known"] for record in records)
    print(f"\n{len(records)} captures, {recovered} accepted as known, "
          f"threshold = {predictor.threshold:.4f}, feature dim = {len(records[0]['feature'])}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([to_jsonable(r) for r in records],
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
