from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

IDGDatasetName = Literal["dsprites", "idsprites", "mpi3d"]
IDGSplitName = Literal["random", "composition", "interpolation", "extrapolation"]
IDGMode = Literal["train", "test"]


def _default_factor_names(dataset: IDGDatasetName, n_factors: int) -> List[str]:
    if dataset in ("dsprites", "idsprites"):
        base5 = ["shape", "scale", "rotation", "pos_x", "pos_y"]
        base6 = ["color", "shape", "scale", "rotation", "pos_x", "pos_y"]
        if n_factors == 6:
            return base6
        if n_factors == 5:
            return base5
        return [f"factor_{i}" for i in range(n_factors)]

    if dataset == "mpi3d":
        base7 = [
            "object_color",
            "object_shape",
            "object_size",
            "camera_height",
            "background_color",
            "robot_arm_horizontal",
            "robot_arm_vertical",
        ]
        if n_factors == 7:
            return base7
        return [f"factor_{i}" for i in range(n_factors)]

    raise ValueError(f"Dataset no soportado: {dataset}")


def _load_npz_array(npz_path: Union[str, Path], prefer_keys: Sequence[str]) -> np.ndarray:
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"No existe: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        for k in prefer_keys:
            if k in data.files:
                return data[k]
        return data[data.files[0]]


def _resolve_file(root: Union[str, Path], dataset: str, filename: str) -> Path:
    root = Path(root)
    p1 = root / filename
    if p1.exists():
        return p1
    p2 = root / dataset / filename
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"No encontré '{filename}' en '{root}' ni en '{root / dataset}'. "
        "(Asegúrate de descargar los .npz con ese nombre.)"
    )


def idg_filename(dataset: IDGDatasetName, split: IDGSplitName, mode: IDGMode, kind: Literal["images", "labels"]) -> str:
    if dataset == "idsprites":
        dataset = "dsprites"
    return f"{dataset}_{split}_{mode}_{kind}.npz"


class IDGBenchmarkDataset(Dataset):
    """
    __getitem__ -> (image, latents, factor_names)
      - image: torch.FloatTensor [C,H,W] en [0,1]
      - latents: torch.LongTensor [F]
      - factor_names: List[str]
    """

    def __init__(
        self,
        root: Union[str, Path],
        dataset: IDGDatasetName,
        split: IDGSplitName,
        mode: IDGMode,
        image_as_float: bool = True,
        latents_dtype: torch.dtype = torch.long,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        image_key_candidates: Sequence[str] = ("images", "x", "arr_0"),
        label_key_candidates: Sequence[str] = ("labels", "y", "arr_0"),
    ) -> None:
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.image_as_float = image_as_float
        self.latents_dtype = latents_dtype
        self.transform = transform

        img_file = idg_filename(dataset, split, mode, "images")
        lab_file = idg_filename(dataset, split, mode, "labels")

        disk_dataset = "dsprites" if dataset == "idsprites" else dataset
        img_path = _resolve_file(self.root, disk_dataset, img_file)
        lab_path = _resolve_file(self.root, disk_dataset, lab_file)

        self._images = _load_npz_array(img_path, image_key_candidates)
        self._labels = _load_npz_array(lab_path, label_key_candidates)

        if len(self._images) != len(self._labels):
            raise ValueError(f"Len mismatch: images={len(self._images)} labels={len(self._labels)}")

        n_factors = 1 if self._labels.ndim == 1 else self._labels.shape[1]
        self.factor_names: List[str] = _default_factor_names(dataset, n_factors)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def _to_chw_tensor(self, img: np.ndarray) -> torch.Tensor:
        if img.ndim == 2:
            t = torch.from_numpy(img)[None, ...]
        elif img.ndim == 3:
            if img.shape[0] in (1, 3) and img.shape[1] != img.shape[0]:
                t = torch.from_numpy(img)
            else:
                t = torch.from_numpy(img).permute(2, 0, 1)
        else:
            raise ValueError(f"Forma de imagen no soportada: {img.shape}")

        if t.shape[0] == 1:
            t = t.repeat(3, 1, 1)

        if self.image_as_float:
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            if t.max() > 1.5:
                t = t / 255.0
        return t

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        img_np = self._images[idx]
        lab_np = self._labels[idx]

        img = self._to_chw_tensor(img_np)
        if self.transform is not None:
            img = self.transform(img)

        latents = torch.as_tensor(lab_np, dtype=self.latents_dtype)
        return img, latents, self.factor_names


def idg_collate_fn(batch: Sequence[Tuple[torch.Tensor, torch.Tensor, List[str]]]):
    imgs, lats, names = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    lats = torch.stack(lats, dim=0) if lats[0].ndim > 0 else torch.tensor(lats)
    factor_names = names[0]
    return imgs, lats, factor_names


@dataclass
class IDGDataConfig:
    root: Union[str, Path]
    dataset: IDGDatasetName
    split: IDGSplitName

    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    make_val: bool = True
    val_fraction: float = 0.1
    val_seed: int = 0

    image_as_float: bool = True
    latents_dtype: torch.dtype = torch.long

    shuffle_train: bool = True
    drop_last_train: bool = True


def make_idg_dataloaders(
    cfg: IDGDataConfig,
    build: Iterable[Literal["train", "val", "test"]] = ("train", "val", "test"),
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, DataLoader]:
    build = set(build)
    out: Dict[str, DataLoader] = {}

    train_ds_full = IDGBenchmarkDataset(
        root=cfg.root,
        dataset=cfg.dataset,
        split=cfg.split,
        mode="train",
        image_as_float=cfg.image_as_float,
        latents_dtype=cfg.latents_dtype,
        transform=transform,
    )

    if "train" in build or "val" in build:
        if cfg.make_val and ("val" in build):
            if not (0.0 < cfg.val_fraction < 1.0):
                raise ValueError("val_fraction debe estar en (0,1) si make_val=True")

            n_total = len(train_ds_full)
            n_val = max(1, int(round(n_total * cfg.val_fraction)))
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(cfg.val_seed)
            train_ds, val_ds = random_split(train_ds_full, [n_train, n_val], generator=gen)

            if "train" in build:
                out["train"] = DataLoader(
                    train_ds,
                    batch_size=cfg.batch_size,
                    shuffle=cfg.shuffle_train,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=cfg.drop_last_train,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )

            if "val" in build:
                out["val"] = DataLoader(
                    val_ds,
                    batch_size=cfg.batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=False,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )
        else:
            if "train" in build:
                out["train"] = DataLoader(
                    train_ds_full,
                    batch_size=cfg.batch_size,
                    shuffle=cfg.shuffle_train,
                    num_workers=cfg.num_workers,
                    pin_memory=cfg.pin_memory,
                    drop_last=cfg.drop_last_train,
                    collate_fn=idg_collate_fn,
                    persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
                )

    if "test" in build:
        test_ds = IDGBenchmarkDataset(
            root=cfg.root,
            dataset=cfg.dataset,
            split=cfg.split,
            mode="test",
            image_as_float=cfg.image_as_float,
            latents_dtype=cfg.latents_dtype,
            transform=transform,
        )
        out["test"] = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
            collate_fn=idg_collate_fn,
            persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        )

    return out
