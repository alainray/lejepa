from __future__ import annotations

import argparse
import json
import math
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import MLP
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import IDGBenchmarkDataset, IDGDatasetName, IDGSplitName

warnings.filterwarnings(
    "ignore",
    message="The epoch parameter in `scheduler.step\\(\\)` was not necessary",
    category=UserWarning,
)


class SICReg(nn.Module):
    def __init__(self, knots: int = 17, t_max: float = 3.0) -> None:
        super().__init__()
        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        proj_dim = proj.size(-1)
        sketch_dim = min(256, proj_dim)
        A = torch.randn(proj_dim, sketch_dim, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ViTEncoder(nn.Module):
    def __init__(self, backbone: str, img_size: int, proj_dim: int) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=512,
            drop_path_rate=0.1,
            img_size=img_size,
        )
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, v = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        proj = self.proj(emb).reshape(n, v, -1).transpose(0, 1)
        return emb, proj


class MultiViewIDGDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset: IDGDatasetName,
        split: IDGSplitName,
        mode: str,
        views: int,
        transform: v2.Compose,
    ) -> None:
        self.views = views
        self.transform = transform
        self.ds = IDGBenchmarkDataset(
            root=root,
            dataset=dataset,
            split=split,
            mode=mode,
            image_as_float=True,
            latents_dtype=torch.long,
            transform=None,
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, latents, _ = self.ds[idx]
        views = torch.stack([self.transform(img) for _ in range(self.views)])
        return views, latents


@dataclass
class ProbeConfig:
    epochs: int
    lr: float
    weight_decay: float
    batch_size: int


def build_transforms(img_size: int, train: bool) -> v2.Compose:
    if train:
        return v2.Compose(
            [
                v2.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.3),
            ]
        )
    return v2.Compose([v2.Resize(img_size), v2.CenterCrop(img_size)])


def seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class JsonlLogger:
    def __init__(self, path: Path, base_fields: Dict[str, object]) -> None:
        self.path = path
        self.base_fields = base_fields
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields: object) -> None:
        payload = {"timestamp": time.time(), **self.base_fields, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento LeJEPA en MPI3D / dSprites")
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dsprites", choices=["dsprites", "mpi3d", "idsprites"])
    parser.add_argument("--split", type=str, default="random", choices=["random", "composition", "interpolation", "extrapolation"])
    parser.add_argument("--backbone", type=str, default="vit_small_patch8_224")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--lamb", type=float, default=0.02)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--probe-weight-decay", type=float, default=1e-6)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="outputs/lejepa_idg")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--method", type=str, default="LeJEPA")
    return parser.parse_args()


def train_lejepa(
    model: ViTEncoder,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: GradScaler,
    device: torch.device,
    lamb: float,
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    sicreg = SICReg().to(device)
    running = {"inv": 0.0, "sicreg": 0.0, "lejepa": 0.0}

    progress = tqdm(train_loader, desc="LeJEPA", leave=False)
    for views, _ in progress:
        views = views.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            _, proj = model(views)
            inv_loss = (proj.mean(0) - proj).square().mean()
            sicreg_loss = sicreg(proj)
            lejepa_loss = sicreg_loss * lamb + inv_loss * (1 - lamb)

        scaler.scale(lejepa_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running["inv"] += inv_loss.item()
        running["sicreg"] += sicreg_loss.item()
        running["lejepa"] += lejepa_loss.item()

        progress.set_postfix({k: f"{v / (progress.n + 1):.4f}" for k, v in running.items()})

    steps = max(1, len(train_loader))
    return {k: v / steps for k, v in running.items()}


def infer_factor_info(dataset: IDGBenchmarkDataset) -> List[int]:
    labels = dataset._labels
    if labels.ndim == 1:
        labels = labels[:, None]
    factors = []
    for i in range(labels.shape[1]):
        factors.append(int(labels[:, i].max()) + 1)
    return factors


def train_probes(
    model: ViTEncoder,
    train_loader: DataLoader,
    test_loader: DataLoader,
    factor_dims: List[int],
    device: torch.device,
    cfg: ProbeConfig,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()
    input_dim = 512
    probes = nn.ModuleList([nn.Linear(input_dim, n) for n in factor_dims]).to(device)
    optimizer = torch.optim.AdamW(probes.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    for epoch in range(cfg.epochs):
        probes.train()
        progress = tqdm(train_loader, desc=f"Probes {epoch + 1}/{cfg.epochs}", leave=False)
        total_loss = 0.0
        factor_loss = [0.0 for _ in factor_dims]
        for imgs, latents, _ in progress:
            imgs = imgs.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                emb, _ = model(imgs[:, None])
            with autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                losses = [F.cross_entropy(head(emb), latents[:, i]) for i, head in enumerate(probes)]
                loss = sum(losses)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            for i, item in enumerate(losses):
                factor_loss[i] += item.item()
            progress.set_postfix(loss=total_loss / (progress.n + 1))
        steps = max(1, len(train_loader))
        epoch_metrics = {"loss_total": total_loss / steps}
        for i, value in enumerate(factor_loss):
            epoch_metrics[f"loss_factor_{i}"] = value / steps
        history.append(epoch_metrics)

    probes.eval()
    correct = [0 for _ in factor_dims]
    total = 0
    with torch.inference_mode():
        for imgs, latents, _ in tqdm(test_loader, desc="Eval probes", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            latents = latents.to(device, non_blocking=True)
            emb, _ = model(imgs[:, None])
            for i, head in enumerate(probes):
                pred = head(emb).argmax(dim=1)
                correct[i] += (pred == latents[:, i]).sum().item()
            total += latents.size(0)

    test_accs = {f"factor_{i}": correct[i] / total for i in range(len(factor_dims))}
    return {"train_history": history, "test_accs": test_accs}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"
    num_gpus = torch.cuda.device_count() if use_cuda else 0
    if num_gpus > 1:
        device = torch.device("cuda:0")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / "metrics.jsonl"
    hparams = {
        "data_root": args.data_root,
        "dataset": args.dataset,
        "split": args.split,
        "backbone": args.backbone,
        "img_size": args.img_size,
        "proj_dim": args.proj_dim,
        "views": args.views,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lamb": args.lamb,
        "num_workers": args.num_workers,
        "amp": args.amp,
        "probe_epochs": args.probe_epochs,
        "probe_lr": args.probe_lr,
        "probe_weight_decay": args.probe_weight_decay,
        "probe_batch_size": args.probe_batch_size,
    }
    logger = JsonlLogger(
        log_path,
        {
            "method": args.method,
            "seed": args.seed,
            "model": args.backbone,
            "hparams": hparams,
        },
    )
    logger.log(event="config")

    train_transform = build_transforms(args.img_size, train=True)
    test_transform = build_transforms(args.img_size, train=False)

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)

    train_ds = MultiViewIDGDataset(
        root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        mode="train",
        views=args.views,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=loader_gen,
        worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed),
    )

    model = ViTEncoder(args.backbone, args.img_size, args.proj_dim).to(device)
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_steps = max(1, len(train_loader))
    total_steps = len(train_loader) * args.epochs
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=1e-4),
        ],
        milestones=[warmup_steps],
    )
    scaler = GradScaler(enabled=args.amp)

    train_start = time.perf_counter()
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        metrics = train_lejepa(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            args.lamb,
            args.amp,
        )
        epoch_time = time.perf_counter() - epoch_start
        logger.log(event="train_epoch", epoch=epoch + 1, metrics=metrics, epoch_time_s=epoch_time)
        summary = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1}/{args.epochs}: {summary}")
    train_time = time.perf_counter() - train_start

    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, save_dir / "lejepa_encoder.pt")

    probe_train_ds = IDGBenchmarkDataset(
        root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        mode="train",
        image_as_float=True,
        latents_dtype=torch.long,
        transform=test_transform,
    )
    probe_test_ds = IDGBenchmarkDataset(
        root=args.data_root,
        dataset=args.dataset,
        split=args.split,
        mode="test",
        image_as_float=True,
        latents_dtype=torch.long,
        transform=test_transform,
    )

    probe_train_loader = DataLoader(
        probe_train_ds,
        batch_size=args.probe_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        generator=loader_gen,
        worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed),
    )
    probe_test_loader = DataLoader(
        probe_test_ds,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=lambda worker_id: seed_worker(worker_id, args.seed),
    )

    factor_dims = infer_factor_info(probe_train_ds)
    probe_cfg = ProbeConfig(
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
    )
    probe_start = time.perf_counter()
    probe_results = train_probes(
        model,
        probe_train_loader,
        probe_test_loader,
        factor_dims,
        device,
        probe_cfg,
        args.amp,
    )
    probe_time = time.perf_counter() - probe_start
    for epoch_idx, metrics in enumerate(probe_results["train_history"], start=1):
        logger.log(event="probe_epoch", epoch=epoch_idx, metrics=metrics)
    logger.log(event="probe_test", metrics=probe_results["test_accs"])
    summary_payload = {
        "train_time_s": train_time,
        "probe_time_s": probe_time,
        "train_epochs": args.epochs,
        "probe_epochs": args.probe_epochs,
        "probe_test": probe_results["test_accs"],
    }
    logger.log(event="run_summary", metrics=summary_payload)
    with (save_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": args.method,
                "seed": args.seed,
                "model": args.backbone,
                "hparams": hparams,
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
