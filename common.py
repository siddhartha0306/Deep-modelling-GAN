from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torch import nn


# src/common.py -> project root is one level above src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

DATASETS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_losses(
    g_losses: list[float],
    d_losses: list[float],
    path: str | Path,
    title: str = "Training losses",
) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(g_losses, label="Generator loss")
    plt.plot(d_losses, label="Discriminator loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] | None = (-1.0, 1.0),
) -> np.ndarray:
    images = images.detach().cpu().float()

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape (N, C, H, W), got {tuple(images.shape)}")

    if normalize:
        if value_range is not None:
            min_v, max_v = value_range
            images = (images - min_v) / (max_v - min_v + 1e-8)
        else:
            min_v = images.min()
            max_v = images.max()
            images = (images - min_v) / (max_v - min_v + 1e-8)

        images = images.clamp(0.0, 1.0)

    n, c, h, w = images.shape
    rows = math.ceil(n / nrow)
    grid = torch.zeros((c, rows * h, nrow * w), dtype=images.dtype)

    for idx in range(n):
        r = idx // nrow
        col = idx % nrow
        grid[:, r * h:(r + 1) * h, col * w:(col + 1) * w] = images[idx]

    grid = grid.permute(1, 2, 0).numpy()

    if grid.shape[-1] == 1:
        grid = grid[..., 0]

    return grid


def save_image_grid(
    images: torch.Tensor,
    path: str | Path,
    nrow: int = 8,
    title: Optional[str] = None,
) -> None:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)

    grid = make_image_grid(images, nrow=nrow, normalize=True, value_range=(-1.0, 1.0))

    plt.figure(figsize=(8, 8))
    if grid.ndim == 2:
        plt.imshow(grid, cmap="gray")
    else:
        plt.imshow(grid)
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


class MLPGenerator(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = latent_dim

        for _ in range(depth):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class MLPDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = in_dim

        for _ in range(depth):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            current_dim = hidden_dim

        layers.extend([
            nn.Linear(current_dim, 1),
            nn.Sigmoid(),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def frechet_distance(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    real_features = np.asarray(real_features, dtype=np.float64)
    fake_features = np.asarray(fake_features, dtype=np.float64)

    mu1 = real_features.mean(axis=0)
    mu2 = fake_features.mean(axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    sigma2 = np.cov(fake_features, rowvar=False)

    eps = 1e-6
    sigma1 = np.atleast_2d(sigma1) + np.eye(np.atleast_2d(sigma1).shape[0]) * eps
    sigma2 = np.atleast_2d(sigma2) + np.eye(np.atleast_2d(sigma2).shape[0]) * eps

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    diff = mu1 - mu2
    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(max(fid, 0.0))