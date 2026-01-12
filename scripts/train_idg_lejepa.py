from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
        self.proj = MLP(512, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

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

    for epoch in range(cfg.epochs):
        probes.train()
        progress = tqdm(train_loader, desc=f"Probes {epoch + 1}/{cfg.epochs}", leave=False)
        total_loss = 0.0
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
            progress.set_postfix(loss=total_loss / (progress.n + 1))

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

    return {f"factor_{i}": correct[i] / total for i in range(len(factor_dims))}


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_transform = build_transforms(args.img_size, train=True)
    test_transform = build_transforms(args.img_size, train=False)

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
    )

    model = ViTEncoder(args.backbone, args.img_size, args.proj_dim).to(device)

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

    for epoch in range(args.epochs):
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
        summary = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"Epoch {epoch + 1}/{args.epochs}: {summary}")

    torch.save(model.state_dict(), save_dir / "lejepa_encoder.pt")

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
    )
    probe_test_loader = DataLoader(
        probe_test_ds,
        batch_size=args.probe_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    factor_dims = infer_factor_info(probe_train_ds)
    probe_cfg = ProbeConfig(
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        batch_size=args.probe_batch_size,
    )
    accs = train_probes(
        model,
        probe_train_loader,
        probe_test_loader,
        factor_dims,
        device,
        probe_cfg,
        args.amp,
    )
    for name, acc in accs.items():
        print(f"Probe {name}: {acc:.4f}")


if __name__ == "__main__":
    main()
