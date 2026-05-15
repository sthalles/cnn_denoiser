"""
Unified training entry point for both the cRM U-Net and the Park R-CED FCN.

Run from repo root, inside the tensorflow-2026 container, e.g.:

    docker exec tensorflow-2026 bash -c \\
      "cd /workspace/cnn_denoiser && \\
       python scripts/training/train.py \\
         --model unet \\
         --tfrecord-root tfrecords_park \\
         --epochs 60 --batch-size 64 \\
         --out-dir models/runs/unet_crm_park_data"

Or for the Park model:

    python scripts/training/train.py --model park --tfrecord-root tfrecords_park ...

Outputs:
  - <out-dir>/best.keras                 (best val_loss checkpoint)
  - <out-dir>/last.keras                 (final epoch)
  - <out-dir>/history.json               (per-epoch metrics)
  - <out-dir>/training_config.json       (hyperparams)
  - <out-dir>/train.log                  (textual log via tee)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import setup_gpu  # noqa: E402
from scripts.data.tfrecord_io import make_crm_dataset, make_mag_dataset  # noqa: E402
from scripts.models.unet_crm import (
    build_unet_crm, crm_compound_loss, mask_mse_metric,
)  # noqa: E402
from scripts.models.park_rced import (
    build_park_rced, log_mag_mse_loss, log_mag_l1_metric,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["unet", "park"], required=True)
    p.add_argument("--tfrecord-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--reduce-lr-patience", type=int, default=4)
    p.add_argument("--reduce-lr-factor", type=float, default=0.5)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def require_gpu() -> None:
    if not tf.config.list_physical_devices("GPU"):
        print("ERROR: GPU not available. Aborting.", file=sys.stderr)
        sys.exit(3)


def build_model_and_loss(args, input_shape):
    if args.model == "unet":
        model = build_unet_crm(input_shape=input_shape)
        loss = crm_compound_loss()
        metrics = [mask_mse_metric()]
    elif args.model == "park":
        model = build_park_rced(input_shape=input_shape)
        loss = log_mag_mse_loss
        metrics = [log_mag_l1_metric()]
    else:
        raise ValueError(args.model)
    return model, loss, metrics


def main() -> None:
    args = parse_args()
    require_gpu()
    setup_gpu()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "training_config.json").write_text(
        json.dumps({k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2)
    )

    # Datasets
    if args.model == "unet":
        train_ds, val_ds, info = make_crm_dataset(
            args.tfrecord_root, batch_size=args.batch_size, repeat=True
        )
        input_shape = (info["n_freq"], info["n_time"], 2)
    else:
        train_ds, val_ds, info = make_mag_dataset(
            args.tfrecord_root, batch_size=args.batch_size, repeat=True
        )
        input_shape = (info["n_freq"], info["n_time"], 1)

    train_examples = int(info["train_examples"])
    val_examples = int(info["val_examples"])
    steps_per_epoch = max(1, train_examples // args.batch_size)
    val_steps = max(1, val_examples // args.batch_size)
    print(f"train examples: {train_examples}, steps/epoch: {steps_per_epoch}")
    print(f"val examples:   {val_examples}, val_steps: {val_steps}")

    # Model
    model, loss, metrics = build_model_and_loss(args, input_shape)
    print(f"Model: {args.model}  params: {model.count_params():,}")

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clipnorm=1.0,
    )
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, jit_compile=False)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(args.out_dir / "best.keras"),
            save_best_only=True, monitor="val_loss", mode="min", verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=args.reduce_lr_patience,
            factor=args.reduce_lr_factor, min_lr=args.min_lr, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.early_stop_patience,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(args.out_dir / "training.csv"), append=False
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    start = time.time()
    history = model.fit(
        train_ds, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds, validation_steps=val_steps,
        callbacks=callbacks, verbose=2,
    )
    elapsed = time.time() - start

    model.save(args.out_dir / "last.keras")
    history_dict = {k: [float(v) for v in vs] for k, vs in history.history.items()}
    history_dict["elapsed_sec"] = elapsed
    (args.out_dir / "history.json").write_text(json.dumps(history_dict, indent=2))
    print(f"DONE in {elapsed/60:.1f} min. best.keras + last.keras saved to {args.out_dir}")


if __name__ == "__main__":
    main()
