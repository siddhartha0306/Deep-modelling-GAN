from __future__ import annotations
import sys
from pathlib import Path
if __package__ is None or __package__ == '':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import (
    ARTIFACTS_DIR,
    DATASETS_DIR,
    MLPDiscriminator,
    MLPGenerator,
    ensure_dir,
    get_device,
    plot_losses,
    save_json,
    set_seed,
)


def load_data(csv_path: str, max_rows: int = 50000):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    if 'Label' not in df.columns:
        raise ValueError('Label column not found in CSV.')
    labels = df['Label'].astype(str).str.upper()
    mask = labels.str.contains('BENIGN') | labels.str.contains('DOS') | labels.str.contains('DDOS')
    df = df.loc[mask].copy(); labels = labels.loc[df.index]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=42); labels = labels.loc[df.index]
    numeric_df = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
    features = StandardScaler().fit_transform(numeric_df.to_numpy(dtype=np.float32)).astype(np.float32)
    return features, labels.to_numpy(), numeric_df.columns.tolist()


def plot_pca(real_data: np.ndarray, fake_data: np.ndarray, save_path):
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(np.vstack([real_data, fake_data]))
    real_proj = reduced[:len(real_data)]
    fake_proj = reduced[len(real_data):]
    plt.figure(figsize=(7, 6))
    plt.scatter(real_proj[:, 0], real_proj[:, 1], s=8, alpha=0.5, label='Real')
    plt.scatter(fake_proj[:, 0], fake_proj[:, 1], s=8, alpha=0.5, label='Fake')
    plt.title('Real vs Fake Traffic Samples (PCA)')
    plt.legend(); plt.tight_layout(); plt.savefig(save_path, dpi=180); plt.close()


def train(args):
    set_seed(42)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)
    features, labels, feature_names = load_data(args.input_csv, args.max_rows)
    loader = DataLoader(TensorDataset(torch.tensor(features)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    input_dim = features.shape[1]
    generator = MLPGenerator(latent_dim=args.latent_dim, out_dim=input_dim, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    discriminator = MLPDiscriminator(in_dim=input_dim, hidden_dim=args.hidden_dim, depth=args.depth).to(device)
    criterion = nn.BCELoss()
    generator_opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    g_losses, d_losses = [], []

    for epoch in range(args.epochs):
        for (real_batch,) in tqdm(loader, desc=f'Epoch {epoch + 1}/{args.epochs}'):
            real_batch = real_batch.to(device)
            bs = real_batch.size(0)
            real_targets = torch.ones(bs, 1, device=device)
            fake_targets = torch.zeros(bs, 1, device=device)

            noise = torch.randn(bs, args.latent_dim, device=device)
            fake_batch = generator(noise).detach()
            d_loss = criterion(discriminator(real_batch), real_targets) + criterion(discriminator(fake_batch), fake_targets)
            discriminator_opt.zero_grad(); d_loss.backward(); discriminator_opt.step()

            noise = torch.randn(bs, args.latent_dim, device=device)
            g_loss = criterion(discriminator(generator(noise)), real_targets)
            generator_opt.zero_grad(); g_loss.backward(); generator_opt.step()

            g_losses.append(float(g_loss.item())); d_losses.append(float(d_loss.item()))

    plot_losses(g_losses, d_losses, output_dir / 'losses.png', title='CICIDS GAN losses')
    with torch.no_grad():
        sample_size = min(2000, len(features))
        real_sample = features[:sample_size]
        fake_sample = generator(torch.randn(sample_size, args.latent_dim, device=device)).cpu().numpy()
    plot_pca(real_sample, fake_sample, output_dir / 'pca_real_vs_fake.png')

    real_stats = pd.DataFrame(real_sample, columns=feature_names).describe().T[['mean', 'std']]
    fake_stats = pd.DataFrame(fake_sample, columns=feature_names).describe().T[['mean', 'std']]
    comparison = real_stats.join(fake_stats, lsuffix='_real', rsuffix='_fake')
    comparison['abs_mean_diff'] = (comparison['mean_real'] - comparison['mean_fake']).abs()
    comparison.sort_values('abs_mean_diff', ascending=False).head(25).to_csv(output_dir / 'feature_stat_comparison_top25.csv')

    save_json({
        'n_rows_used': int(features.shape[0]),
        'n_features': int(features.shape[1]),
        'final_g_loss': g_losses[-1],
        'final_d_loss': d_losses[-1],
        'class_counts': pd.Series(labels).value_counts().to_dict(),
    }, output_dir / 'metrics.json')
    torch.save(generator.state_dict(), output_dir / 'generator.pt')
    torch.save(discriminator.state_dict(), output_dir / 'discriminator.pt')


def main():
    parser = argparse.ArgumentParser(description='CICIDS tabular GAN')
    parser.add_argument('--input_csv', type=str, default=str(DATASETS_DIR / 'cicids' / 'Tuesday-WorkingHours.pcap_ISCX.csv'))
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--max_rows', type=int, default=50000)
    parser.add_argument('--output_dir', type=str, default=str(ARTIFACTS_DIR / 'cicids'))
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
