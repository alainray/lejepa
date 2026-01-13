from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch

from dataset import IDGBenchmarkDataset
from scripts.train_idg_lejepa import (
    JsonlLogger,
    ProbeConfig,
    ViTEncoder,
    build_transforms,
    infer_factor_info,
    seed_worker,
    set_seed,
    train_probes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-ejecutar probes IDG desde un checkpoint guardado.")
    parser.add_argument("--run-dir", type=str, required=True, help="Directorio del experimento con summary.json.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", help="Forzar AMP incluso si no está en hparams.")
    parser.add_argument("--probe-epochs", type=int, default=None)
    parser.add_argument("--probe-lr", type=float, default=None)
    parser.add_argument("--probe-weight-decay", type=float, default=None)
    parser.add_argument("--probe-batch-size", type=int, default=None)
    return parser.parse_args()


def load_summary(run_dir: Path) -> Dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No encontré summary.json en: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    summary = load_summary(run_dir)

    hparams = summary.get("hparams", {})
    if not hparams:
        raise ValueError("summary.json no contiene hparams.")

    set_seed(int(summary.get("seed", 0)))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible, usando CPU.")
        device = torch.device("cpu")

    use_amp = bool(args.amp or hparams.get("amp", False))

    img_size = int(hparams["img_size"])
    train_transform = build_transforms(img_size, train=False)

    probe_train_ds = IDGBenchmarkDataset(
        root=hparams["data_root"],
        dataset=hparams["dataset"],
        split=hparams["split"],
        mode="train",
        image_as_float=True,
        latents_dtype=torch.long,
        transform=train_transform,
    )
    probe_test_ds = IDGBenchmarkDataset(
        root=hparams["data_root"],
        dataset=hparams["dataset"],
        split=hparams["split"],
        mode="test",
        image_as_float=True,
        latents_dtype=torch.long,
        transform=train_transform,
    )

    loader_gen = torch.Generator().manual_seed(int(summary.get("seed", 0)))
    num_workers = int(hparams.get("num_workers", 0))

    batch_size = args.probe_batch_size or int(hparams["probe_batch_size"])
    probe_train_loader = torch.utils.data.DataLoader(
        probe_train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=loader_gen,
        worker_init_fn=lambda worker_id: seed_worker(worker_id, int(summary.get("seed", 0))),
    )
    probe_test_loader = torch.utils.data.DataLoader(
        probe_test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=lambda worker_id: seed_worker(worker_id, int(summary.get("seed", 0))),
    )

    model = ViTEncoder(
        hparams["backbone"],
        img_size,
        int(hparams["proj_dim"]),
    ).to(device)
    checkpoint_path = run_dir / "lejepa_encoder.pt"
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    factor_dims = infer_factor_info(probe_train_ds, probe_test_ds)
    probe_cfg = ProbeConfig(
        epochs=args.probe_epochs or int(hparams["probe_epochs"]),
        lr=args.probe_lr or float(hparams["probe_lr"]),
        weight_decay=args.probe_weight_decay or float(hparams["probe_weight_decay"]),
        batch_size=batch_size,
    )

    logger = JsonlLogger(
        run_dir / "metrics.jsonl",
        {
            "method": summary.get("method"),
            "seed": summary.get("seed"),
            "model": summary.get("model"),
            "hparams": hparams,
        },
    )
    logger.log(event="probe_rerun_config", metrics={"probe_cfg": probe_cfg.__dict__})

    probe_start = time.perf_counter()
    probe_results = train_probes(
        model,
        probe_train_loader,
        probe_test_loader,
        factor_dims,
        device,
        probe_cfg,
        use_amp,
    )
    probe_time = time.perf_counter() - probe_start

    for epoch_idx, metrics in enumerate(probe_results["train_history"], start=1):
        logger.log(event="probe_epoch", epoch=epoch_idx, metrics=metrics)
    logger.log(event="probe_test", metrics=probe_results["test_accs"])

    summary_payload = {
        "train_time_s": summary.get("train_time_s"),
        "probe_time_s": probe_time,
        "train_epochs": summary.get("train_epochs"),
        "probe_epochs": probe_cfg.epochs,
        "probe_test": probe_results["test_accs"],
    }
    logger.log(event="run_summary", metrics=summary_payload)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": summary.get("method"),
                "seed": summary.get("seed"),
                "model": summary.get("model"),
                "hparams": hparams,
                "train_history": summary.get("train_history", []),
                **summary_payload,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    for name, acc in probe_results["test_accs"].items():
        print(f"Probe {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
