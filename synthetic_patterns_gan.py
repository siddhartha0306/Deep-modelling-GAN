from __future__ import annotations
import sys
from pathlib import Path
if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    ARTIFACTS_DIR,
    MLPDiscriminator,
    MLPGenerator,
    ensure_dir,
    get_device,
    plot_losses,
    save_json,
    set_seed,
)


def build_sine_wave(n: int = 3000) -> np.ndarray:
    x = np.random.uniform(-np.pi, np.pi, size=n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    return np.column_stack([x, y]).astype(np.float32)


def build_spiral(n: int = 3000) -> np.ndarray:
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi
    r = 0.5 * theta
    x = r * np.cos(theta) + 0.1 * np.random.randn(n)
    y = r * np.sin(theta) + 0.1 * np.random.randn(n)
    data = np.column_stack([x, y]).astype(np.float32)
    return data / np.std(data, axis=0, keepdims=True)


def build_gaussian_mixture(n: int = 3000) -> np.ndarray:
    centers = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=np.float32)
    points = []
    per_cluster = n // len(centers)
    for center in centers:
        points.append(center + 0.4 * np.random.randn(per_cluster, 2))
    return np.vstack(points).astype(np.float32)


def build_noisy_curve(n: int = 3000) -> np.ndarray:
    x = np.random.uniform(-3.0, 3.0, size=n)
    eps = np.random.normal(0, 0.15, size=n)
    y = np.sin(2 * x) + 0.3 * np.cos(5 * x) + eps
    return np.column_stack([x, y]).astype(np.float32)


def get_dataset(name: str, n: int = 3000) -> np.ndarray:
    builders = {
        'sine': build_sine_wave,
        'spiral': build_spiral,
        'gaussian_mixture': build_gaussian_mixture,
        'noisy_curve': build_noisy_curve,
    }
    return builders[name](n)


def train_gan(data: np.ndarray, output_dir: Path, latent_dim: int = 8, hidden_dim: int = 128, depth: int = 2, epochs: int = 300, batch_size: int = 128, lr: float = 2e-4) -> None:
    device = get_device()
    loader = DataLoader(TensorDataset(torch.tensor(data)), batch_size=batch_size, shuffle=True, drop_last=True)

    generator = MLPGenerator(latent_dim=latent_dim, out_dim=2, hidden_dim=hidden_dim, depth=depth).to(device)
    discriminator = MLPDiscriminator(in_dim=2, hidden_dim=hidden_dim, depth=depth).to(device)
    criterion = nn.BCELoss()
    generator_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []
    for _ in tqdm(range(epochs), desc='Training synthetic GAN'):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)
            batch_size_now = real_batch.size(0)
            real_targets = torch.ones(batch_size_now, 1, device=device)
            fake_targets = torch.zeros(batch_size_now, 1, device=device)

            z = torch.randn(batch_size_now, latent_dim, device=device)
            fake_batch = generator(z).detach()
            d_loss = criterion(discriminator(real_batch), real_targets) + criterion(discriminator(fake_batch), fake_targets)
            discriminator_opt.zero_grad(); d_loss.backward(); discriminator_opt.step()

            z = torch.randn(batch_size_now, latent_dim, device=device)
            g_loss = criterion(discriminator(generator(z)), real_targets)
            generator_opt.zero_grad(); g_loss.backward(); generator_opt.step()

            g_losses.append(float(g_loss.item()))
            d_losses.append(float(d_loss.item()))

    plot_losses(g_losses, d_losses, output_dir / 'losses.png', title='Synthetic GAN losses')
    generator.eval()
    with torch.no_grad():
        samples = generator(torch.randn(len(data), latent_dim, device=device)).cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.scatter(data[:, 0], data[:, 1], s=6, alpha=0.5); plt.title('Real data')
    plt.subplot(1, 2, 2); plt.scatter(samples[:, 0], samples[:, 1], s=6, alpha=0.5); plt.title('Generated data')
    plt.tight_layout(); plt.savefig(output_dir / 'real_vs_fake.png', dpi=180); plt.close()

    save_json({'epochs': epochs, 'final_g_loss': g_losses[-1], 'final_d_loss': d_losses[-1]}, output_dir / 'metrics.json')
    torch.save(generator.state_dict(), output_dir / 'generator.pt')
    torch.save(discriminator.state_dict(), output_dir / 'discriminator.pt')


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic 2D GAN')
    parser.add_argument('--dataset', type=str, default='noisy_curve', choices=['sine', 'spiral', 'gaussian_mixture', 'noisy_curve'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default=str(ARTIFACTS_DIR / 'synthetic_patterns'))
    args = parser.parse_args()
    set_seed(42)
    out = ensure_dir(args.output_dir)
    train_gan(get_dataset(args.dataset), out, args.latent_dim, args.hidden_dim, args.depth, args.epochs, args.batch_size)


if __name__ == '__main__':
    main()
