"""
Supplementary Figure 17: regenerate the t-SNE comparison from the saved WAE weights
and export the plotted coordinates as Source Data.

The WAE is NOT retrained. The weights in WAE_model_weights_pw.pth are loaded as they are,
so the model used in the paper is unchanged.
Put this script in the same folder as train_wae.py and run:
    python wae_tsne_export.py
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from train_wae import load_data, WAE, ELEMENTS, DATA_PATH, OUT_PATH, SCRIPT_DIR, device

SEED  = 42
N_GEN = 200


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

    comp_np = load_data(DATA_PATH)
    model = WAE(input_size=comp_np.shape[1]).to(device)
    model.load_state_dict(torch.load(OUT_PATH, map_location=device))
    model.eval()
    print(f"Loaded WAE weights from {OUT_PATH}")

    with torch.no_grad():
        x = torch.FloatTensor(comp_np).to(device)
        x_recon, _ = model(x)
        recon_np = x_recon.cpu().numpy()
        z_samples = torch.randn(N_GEN, 7).to(device)
        gen_np = model.decoder(z_samples).cpu().numpy()

    all_comp = np.vstack([comp_np, recon_np, gen_np])
    n_orig, n_recon = len(comp_np), len(recon_np)
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=min(30, len(all_comp) - 1))
    coords = tsne.fit_transform(all_comp)

    # ── Source Data ──
    groups = (['Original'] * n_orig + ['Reconstructed'] * n_recon
              + ['Generated'] * len(gen_np))
    panels = (['a, b'] * n_orig + ['a'] * n_recon + ['b'] * len(gen_np))
    df = pd.DataFrame(all_comp, columns=ELEMENTS)
    df.insert(0, 'tsne_2', coords[:, 1])
    df.insert(0, 'tsne_1', coords[:, 0])
    df.insert(0, 'group', groups)
    df.insert(0, 'panel', panels)
    out_csv = os.path.join(SCRIPT_DIR, 'wae_tsne_coordinates.csv')
    df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"Source data saved -> {out_csv}")

    # ── Figure (same style as train_wae.py) ──
    matplotlib.rcParams.update({'font.family': ['Arial', 'sans-serif'],
                                'font.size': 8, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 6/2.54),
                             gridspec_kw={'wspace': 0.4})

    ax = axes[0]
    ax.scatter(coords[:n_orig, 0], coords[:n_orig, 1],
               c='#E64B35', s=20, alpha=0.8, label='Original', edgecolors='none')
    ax.scatter(coords[n_orig:n_orig+n_recon, 0], coords[n_orig:n_orig+n_recon, 1],
               c='#4DBBD5', s=20, alpha=0.6, label='Reconstructed', edgecolors='none')
    ax.set_title('Original vs Reconstructed', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('t-SNE 1', fontsize=7); ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.text(-0.18, 1.08, 'a', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    ax = axes[1]
    ax.scatter(coords[:n_orig, 0], coords[:n_orig, 1],
               c='#E64B35', s=20, alpha=0.8, label='Original', edgecolors='none')
    ax.scatter(coords[n_orig+n_recon:, 0], coords[n_orig+n_recon:, 1],
               c='#00A087', s=20, alpha=0.6, label='Generated', edgecolors='none')
    ax.set_title('Original vs Generated', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('t-SNE 1', fontsize=7); ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.text(-0.18, 1.08, 'b', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(SCRIPT_DIR, f'wae_tsne_comparison_export.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Figure saved -> {SCRIPT_DIR}/wae_tsne_comparison_export.png")
    print(f"n_original = {n_orig}, n_reconstructed = {n_recon}, n_generated = {len(gen_np)}")


if __name__ == '__main__':
    main()