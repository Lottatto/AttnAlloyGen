import os, re, random, warnings, argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 避免多OpenMP库冲突警告
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.core.periodic_table import Element

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, 'database_colab_text.xlsx')
OUT_PATH   = os.path.join(SCRIPT_DIR, 'WAE_model_weights_pw.pth')

# ── 常量 ──────────────────────────────────────────────────────
ELEMENTS = ['Mg','Al','Zn','Li','Y','Mn','Nd','Ce','Ca','Gd','Zr','Sc','Fe','Sn','In','La']
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'

EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0','10.1007/s11665-026-13708-2',
    '10.1016/j.jma.2018.11.004', '10.1016/j.jmrt.2022.05.161',
}

# ── 训练超参数（和原WAE_update.py保持一致）─────────────────────
DEFAULT_PARAMS = {
    'num_epoch':   2300,
    'batch_size':  32,
    'lr':          5e-5,
    'sigma':       10,
    'MMD_lambda':  1e-2,
    'weight_decay':1e-5,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 1. 数据加载（与 generation_quality.py 完全一致）
# ══════════════════════════════════════════════════════════════
def _std_immersion(t):
    t = str(t).strip().lower()
    for p, fn in [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?', lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?',  lambda m: f"{int(float(m.group(1))*24)} hours"),
    ]:
        m = re.search(p, t)
        if m: return fn(m)
    return t

def _std_ht(t):
    t = str(t).strip().lower()
    for p, r in [
        (r'\bt4\b','solution-treated'),(r'\bt6\b','solution-treated+aging'),
        (r'solution\s*treat','solution-treated'),(r'anneal','annealed'),
        (r'aging|aged','aging'),
        (r'no\s*heat\s*treat|as[\s-]?cast','no heat treatment'),
    ]:
        if re.search(p, t): return r
    return t

def _std_temp(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I):
        return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', t)
    return '+'.join([f"{float(x):.0f}°C" for x in m]) if m else t

def _std_time(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I):
        return 'No heat treatment time'
    def _n(s):
        s = str(s).strip().lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*h', s)
        if m: return f"{float(m.group(1)):.0f} hours"
        m = re.search(r'(\d+(?:\.\d+)?)\s*min', s)
        return f"{float(m.group(1)):.0f} min" if m else s
    return '+'.join([_n(p) for p in re.split(r'[+,;/]', t) if p.strip()])

def _std_deform(t):
    t = str(t).strip().lower()
    for p, r in [
        (r'cast(?:ed|ing)?|as[\s-]?cast','cast'),
        (r'roll(?:ed|ing)?','rolling'),
        (r'extrud(?:ed|ing)|extrusion','extruded'),
    ]:
        if re.search(p, t, re.I): return r
    return t

def load_data(path=DATA_PATH):
    """加载并过滤数据，返回完整 df 和归一化的成分矩阵"""
    df = pd.read_excel(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            mv = df[col].mode()
            df[col] = df[col].fillna(mv[0] if not mv.empty else 'unknown')
        else:
            try: df[col] = df[col].fillna(df[col].median())
            except: pass

    # 与 generation_quality.py 完全一致的过滤顺序
    df = df[df[TARGET_COL] <= 30]
    df = df[df[TARGET_COL] <= df[TARGET_COL].mean() + 2.5*df[TARGET_COL].std()]
    df = df[df['Test method'] == 'pw'].copy()

    for col, fn in [
        ('Immersion Time',           _std_immersion),
        ('Heat Treatment Process',   _std_ht),
        ('Heat Treatment Temperature',_std_temp),
        ('Heat Treatment Time',      _std_time),
        ('Deformation Process',      _std_deform),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(fn)

    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df = df.reset_index(drop=True)

    # 归一化成分（原子分数）
    comp = df[ELEMENTS].values.astype(float)
    comp = comp / (comp.sum(axis=1, keepdims=True) + 1e-10)

    print(f"Data loaded: {len(df)} samples (pw, after filtering)")
    cr = df[TARGET_COL]
    print(f"CR: mean={cr.mean():.2f}, std={cr.std():.2f}, "
          f"min={cr.min():.4f}, max={cr.max():.2f} mm/yr")

    # 检查Mg质量分数分布
    _aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
    wt = comp * _aw
    wt = wt / (wt.sum(axis=1, keepdims=True) + 1e-10)
    mg_wt = wt[:, 0]
    n_valid = ((mg_wt >= 0.94) & (mg_wt <= 0.99)).sum()
    print(f"Mg wt%: mean={mg_wt.mean():.3f}, "
          f"in [94,99]%: {n_valid}/{len(comp)} ({n_valid/len(comp):.1%})")

    return comp


# ══════════════════════════════════════════════════════════════
# 2. WAE 模型（与 model.py 结构完全一致）
# ══════════════════════════════════════════════════════════════
class CustomSoftmax(nn.Module):
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = F.softmax(x, dim=1)
        return x / x.sum(dim=1, keepdim=True)

class WAE(nn.Module):
    def __init__(self, input_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Dropout(0.01),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Dropout(0.01),
            nn.Linear(64, 7),
        )
        self.decoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Linear(128, input_size),
            CustomSoftmax(),
        )

    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        z = self.encoder(x)
        return self.decoder(z), z


# ══════════════════════════════════════════════════════════════
# 3. MMD 损失
# ══════════════════════════════════════════════════════════════
def compute_mmd(x, y, sigma_list=(1, 2, 4, 8, 16)):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    def kern(a, b):
        aa = (a * a).sum(1, keepdim=True)
        bb = (b * b).sum(1, keepdim=True)
        d  = aa + bb.t() - 2 * a.mm(b.t())
        K  = torch.zeros_like(d)
        for s in sigma_list:
            K += torch.exp(-0.5 * d / (s * s))
        return K

    Kxx = kern(x, x); Kyy = kern(y, y); Kxy = kern(x, y)
    m, n = x.size(0), y.size(0)
    return Kxx.sum()/(m*m) + Kyy.sum()/(n*n) - 2*Kxy.sum()/(m*n)


# ══════════════════════════════════════════════════════════════
# 4. 训练
# ══════════════════════════════════════════════════════════════
def train_wae(comp_np, params, out_path=OUT_PATH):
    """
    训练 WAE 并保存权重。
    comp_np: (N, 16) 归一化原子分数
    """
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    dataset = TensorDataset(torch.FloatTensor(comp_np))
    loader  = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    model     = WAE(input_size=comp_np.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    losses = {'recon': [], 'mmd': [], 'constraint': [], 'total': []}
    best_loss   = float('inf')
    best_state  = None

    print(f"\nTraining WAE on {len(comp_np)} samples for {params['num_epoch']} epochs...")
    print(f"  batch_size={params['batch_size']}, lr={params['lr']}, "
          f"MMD_lambda={params['MMD_lambda']}")

    for epoch in range(params['num_epoch']):
        model.train()
        ep_recon = ep_mmd = ep_cons = ep_total = 0.0
        n_batch  = 0

        for (x,) in loader:
            x = x.to(device)
            z_prior = torch.randn(x.size(0), 7).to(device)

            x_recon, z = model(x)

            # 重构损失（KL散度，与原 WAE_update.py 一致）
            recon_loss = F.kl_div(
                (x_recon + 1e-8).log(), x + 1e-8, reduction='batchmean')

            # MMD 损失
            mmd_loss = compute_mmd(z, z_prior)

            # 成分约束
            sum_loss = torch.abs(x_recon.sum(dim=1) - 1.0).mean()
            neg_loss = F.relu(-x_recon).mean()

            loss = (recon_loss
                    + params['MMD_lambda'] * mmd_loss
                    + 0.1 * sum_loss
                    + 0.1 * neg_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_recon += recon_loss.item()
            ep_mmd   += mmd_loss.item()
            ep_cons  += (sum_loss + neg_loss).item()
            ep_total += loss.item()
            n_batch  += 1

        ep_total /= n_batch
        losses['recon'].append(ep_recon / n_batch)
        losses['mmd'].append(ep_mmd / n_batch)
        losses['constraint'].append(ep_cons / n_batch)
        losses['total'].append(ep_total)

        if ep_total < best_loss:
            best_loss  = ep_total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch [{epoch+1}/{params['num_epoch']}] "
                  f"Total={ep_total:.6f}  Recon={losses['recon'][-1]:.6f}  "
                  f"MMD={losses['mmd'][-1]:.6f}")

    # 保存最优权重
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_path)
    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Saved → {out_path}")

    return model, losses


# ══════════════════════════════════════════════════════════════
# 5. 可视化训练曲线
# ══════════════════════════════════════════════════════════════
def plot_losses(losses, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12/2.54, 5/2.54),
                              gridspec_kw={'wspace': 0.4})
    matplotlib.rcParams.update({'font.family':['Arial','sans-serif'],
                                 'font.size':8, 'pdf.fonttype':42})

    for ax, (key, label, color) in zip(
        axes,
        [('total', 'Total loss', '#E64B35'),
         ('recon', 'Reconstruction loss', '#4DBBD5')],
    ):
        ax.plot(losses[key], color=color, lw=1.2)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylabel(label, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'wae_training_loss.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Loss curve saved → {out_dir}/wae_training_loss.png")


# ══════════════════════════════════════════════════════════════
# 6. 重构质量验证
# ══════════════════════════════════════════════════════════════
def evaluate_reconstruction(model, comp_np):
    """计算重构MSE和相关系数，验证WAE训练质量"""
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(comp_np).to(device)
        x_recon, _ = model(x)
        recon_np = x_recon.cpu().numpy()

    mse  = np.mean((comp_np - recon_np) ** 2)
    # 逐元素相关系数（只对均值>0的元素）
    corrs = []
    for i, e in enumerate(ELEMENTS):
        if comp_np[:, i].std() > 1e-6:
            c = np.corrcoef(comp_np[:, i], recon_np[:, i])[0, 1]
            if not np.isnan(c):
                corrs.append(c)

    print(f"\nReconstruction quality:")
    print(f"  MSE = {mse:.6f}")
    print(f"  Mean correlation (active elements) = {np.mean(corrs):.4f}")
    return mse, np.mean(corrs)


# ══════════════════════════════════════════════════════════════
# 7. t-SNE可视化：验证生成成分和训练集分布一致
# ══════════════════════════════════════════════════════════════
def plot_tsne_comparison(model, comp_np, out_dir, n_gen=200):
    """
    用t-SNE对比原始训练集和WAE重构/生成成分的分布。
    类似论文Figure S13，验证WAE学到了真实分布而非随机生成。
    """
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(comp_np).to(device)
        x_recon, z_train = model(x)
        recon_np = x_recon.cpu().numpy()
        z_np     = z_train.cpu().numpy()

        # 从latent space采样生成新成分
        z_samples = torch.randn(n_gen, 7).to(device)
        gen_np    = model.decoder(z_samples).cpu().numpy()

    # 合并所有成分用于t-SNE
    all_comp = np.vstack([comp_np, recon_np, gen_np])
    n_orig   = len(comp_np)
    n_recon  = len(recon_np)

    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(all_comp)-1))
    coords = tsne.fit_transform(all_comp)

    matplotlib.rcParams.update({'font.family':['Arial','sans-serif'],
                                 'font.size':8, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(14/2.54, 6/2.54),
                              gridspec_kw={'wspace': 0.4})

    # (a) 原始 vs 重构
    ax = axes[0]
    ax.scatter(coords[:n_orig, 0], coords[:n_orig, 1],
               c='#E64B35', s=20, alpha=0.8, label='Original', edgecolors='none')
    ax.scatter(coords[n_orig:n_orig+n_recon, 0],
               coords[n_orig:n_orig+n_recon, 1],
               c='#4DBBD5', s=20, alpha=0.6, label='Reconstructed', edgecolors='none')
    ax.set_title('Original vs Reconstructed', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('t-SNE 1', fontsize=7); ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.text(-0.18, 1.08, 'a', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    # (b) 原始 vs 生成
    ax = axes[1]
    ax.scatter(coords[:n_orig, 0], coords[:n_orig, 1],
               c='#E64B35', s=20, alpha=0.8, label='Original', edgecolors='none')
    ax.scatter(coords[n_orig+n_recon:, 0],
               coords[n_orig+n_recon:, 1],
               c='#00A087', s=20, alpha=0.6, label='Generated', edgecolors='none')
    ax.set_title('Original vs Generated', fontsize=8, fontweight='bold')
    ax.legend(fontsize=6.5, frameon=False)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xlabel('t-SNE 1', fontsize=7); ax.set_ylabel('t-SNE 2', fontsize=7)
    ax.text(-0.18, 1.08, 'b', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'wae_tsne_comparison.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"t-SNE comparison saved → {out_dir}/wae_tsne_comparison.png")


# ══════════════════════════════════════════════════════════════
# 8. 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',   default=DATA_PATH)
    parser.add_argument('--out',    default=OUT_PATH)
    parser.add_argument('--epochs', type=int, default=DEFAULT_PARAMS['num_epoch'])
    parser.add_argument('--lr',     type=float, default=DEFAULT_PARAMS['lr'])
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params['num_epoch'] = args.epochs
    params['lr']        = args.lr
    out_dir = os.path.dirname(args.out) or SCRIPT_DIR

    print("=" * 55)
    print("Train WAE on pw dataset")
    print(f"  Device : {device}")
    print(f"  Data   : {args.data}")
    print(f"  Output : {args.out}")
    print(f"  Epochs : {params['num_epoch']}")
    print("=" * 55)

    comp_np = load_data(args.data)
    model, losses = train_wae(comp_np, params, out_path=args.out)
    evaluate_reconstruction(model, comp_np)
    plot_losses(losses, out_dir)
    plot_tsne_comparison(model, comp_np, out_dir)

    print("\nDone. Now update generation_quality.py to use:")
    print(f"  WAE_WEIGHTS = '{args.out}'")