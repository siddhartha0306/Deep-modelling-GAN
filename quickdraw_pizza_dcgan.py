from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    ARTIFACTS_DIR,
    DATASETS_DIR,
    ensure_dir,
    frechet_distance,
    get_device,
    plot_losses,
    save_image_grid,
    save_json,
    set_seed,
)


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 64, channels: int = 1, features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features * 2, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, channels: int = 1, features: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, 1)


def weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname or "BatchNorm" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)


def load_quickdraw_npy(input_npy: str, max_samples: int = 10000) -> torch.Tensor:
    array = np.load(input_npy)
    if array.ndim != 2 or array.shape[1] != 28 * 28:
        raise ValueError("Expected a QuickDraw .npy array with shape [N, 784].")

    if len(array) > max_samples:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(array), size=max_samples, replace=False)
        array = array[indices]

    array = array.reshape(-1, 1, 28, 28).astype(np.float32)
    array = (array / 127.5) - 1.0
    tensor = torch.tensor(array, dtype=torch.float32)
    return F.interpolate(tensor, size=(32, 32), mode="bilinear", align_corners=False)


def pca_features(images: torch.Tensor, n_components: int = 32) -> np.ndarray:
    flat = images.detach().cpu().numpy().reshape(images.shape[0], -1)
    n_components = min(n_components, flat.shape[0], flat.shape[1])
    return PCA(n_components=n_components).fit_transform(flat)


def train(args):
    set_seed(args.seed)
    device = get_device()
    out = ensure_dir(args.output_dir)
    print(f"Saving outputs to: {out.resolve()}")

    data = load_quickdraw_npy(args.input_npy, max_samples=args.max_samples)
    loader = DataLoader(
        TensorDataset(data),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=False,
    )

    generator = Generator(
        latent_dim=args.latent_dim,
        features=args.base_features,
    ).to(device)

    discriminator = Discriminator(
        features=args.base_features,
    ).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, args.latent_dim, 1, 1, device=device)

    g_losses: list[float] = []
    d_losses: list[float] = []

    generator.train()
    discriminator.train()

    for epoch in range(args.epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        steps = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for (real,) in progress:
            real = real.to(device)
            bs = real.size(0)

            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            noise = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            fake = generator(noise).detach()

            discriminator_opt.zero_grad(set_to_none=True)
            d_loss = criterion(discriminator(real), real_targets) + criterion(discriminator(fake), fake_targets)
            d_loss.backward()
            discriminator_opt.step()

            noise = torch.randn(bs, args.latent_dim, 1, 1, device=device)
            generator_opt.zero_grad(set_to_none=True)
            g_loss = criterion(discriminator(generator(noise)), real_targets)
            g_loss.backward()
            generator_opt.step()

            g_val = float(g_loss.item())
            d_val = float(d_loss.item())

            g_losses.append(g_val)
            d_losses.append(d_val)

            epoch_g_loss += g_val
            epoch_d_loss += d_val
            steps += 1

            progress.set_postfix(g_loss=f"{g_val:.4f}", d_loss=f"{d_val:.4f}")

        avg_g = epoch_g_loss / max(steps, 1)
        avg_d = epoch_d_loss / max(steps, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} - G: {avg_g:.4f} | D: {avg_d:.4f}")

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise).cpu()
            save_image_grid(
                samples,
                out / f"samples_epoch_{epoch + 1:03d}.png",
                nrow=8,
                title=f"Generated pizza sketches epoch {epoch + 1}",
            )
            generator.train()

    plot_losses(g_losses, d_losses, out / "losses.png", title="QuickDraw pizza DCGAN losses")

    real_batch = data[:64]
    save_image_grid(real_batch, out / "real_grid.png", nrow=8, title="Real pizza sketches")

    generator.eval()
    with torch.no_grad():
        fake_batch = generator(torch.randn(64, args.latent_dim, 1, 1, device=device)).cpu()

    save_image_grid(fake_batch, out / "fake_grid.png", nrow=8, title="Generated pizza sketches")

    fid_real = real_batch[:args.fid_samples]
    fid_fake = fake_batch[:args.fid_samples]

    save_json(
        {
            "n_samples_used": int(data.shape[0]),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "latent_dim": args.latent_dim,
            "final_g_loss": float(g_losses[-1]) if g_losses else None,
            "final_d_loss": float(d_losses[-1]) if d_losses else None,
            "approx_fid_pca": float(
                frechet_distance(
                    pca_features(fid_real, n_components=args.pca_components),
                    pca_features(fid_fake, n_components=args.pca_components),
                )
            ),
        },
        out / "metrics.json",
    )

    if args.save_models:
        torch.save(generator.state_dict(), out / "generator.pt")
        torch.save(discriminator.state_dict(), out / "discriminator.pt")


def main():
    parser = argparse.ArgumentParser(description="Fast QuickDraw pizza DCGAN")
    parser.add_argument("--input_npy", type=str, default=str(DATASETS_DIR / "quickdraw" / "pizza.npy"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--base_features", type=int, default=32)
    parser.add_argument("--fid_samples", type=int, default=64)
    parser.add_argument("--pca_components", type=int, default=32)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(ARTIFACTS_DIR / "quickdraw"))
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()