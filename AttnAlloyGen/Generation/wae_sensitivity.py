import os, sys, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ── 从generation_quality.py导入公共组件 ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from generation_quality import (
    DATA_PATH, WAE_WEIGHTS, ELEMENTS, _aw,
    load_data, atomic_to_wt,
    is_valid, post_process_comp, score_comp, normalize_comp,
    _CBN, _Softmax,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_GEN = 300

# ── 超参数扫描配置 ────────────────────────────────────────────
PARAM_CONFIGS = {
    'latent_dim': {
        'label':   'Latent space dimension',
        'values':  [5, 7, 9],
        'current': 1,
        'tags':    ['dim = 5', 'dim = 7\n(this work)', 'dim = 9'],
    },
    'mcmc_sigma': {
        'label':   'MCMC step size (σ)',
        'values':  [
            [0.005, 0.01, 0.02],
            [0.01,  0.02, 0.03, 0.04],
            [0.02,  0.05, 0.08],
        ],
        'current': 1,
        'tags':    ['small\n[0.005–0.02]', 'medium\n[0.01–0.04]\n(this work)', 'large\n[0.02–0.08]'],
    },
    'mg_constraint': {
        'label':   'Mg content constraint',
        'values':  [(0.88, 0.99), (0.90, 0.99), (0.92, 0.98)],
        'current': 1,
        'tags':    ['loose\n[88–99%]', 'standard\n[90–99%]\n(this work)', 'strict\n[92–98%]'],
    },
}


# ══════════════════════════════════════════════════════════════
# 1. WAE（支持可变latent_dim）
# ══════════════════════════════════════════════════════════════
class WAE(nn.Module):
    def __init__(self, input_size=16, latent_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01),
            nn.Linear(128, 64),  nn.BatchNorm1d(64,  track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01),
            nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),  nn.BatchNorm1d(64,  track_running_stats=False), nn.LeakyReLU(0.0),
            nn.Linear(64, 128),         nn.BatchNorm1d(128, track_running_stats=False), nn.LeakyReLU(0.0),
            nn.Linear(128, input_size), _Softmax())
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        z = self.encoder(x); return self.decoder(z), z

def _replace_bn(m):
    for name, child in m.named_children():
        if isinstance(child, nn.BatchNorm1d): setattr(m, name, _CBN())
        else: _replace_bn(child)

def compute_mmd(x, y, sigma_list=(1,2,4,8,16)):
    x = x.view(x.size(0),-1); y = y.view(y.size(0),-1)
    def kern(a,b):
        aa=(a*a).sum(1,keepdim=True); bb=(b*b).sum(1,keepdim=True)
        d=aa+bb.t()-2*a.mm(b.t()); K=torch.zeros_like(d)
        for s in sigma_list: K+=torch.exp(-0.5*d/(s*s))
        return K
    Kxx=kern(x,x); Kyy=kern(y,y); Kxy=kern(x,y)
    m,n=x.size(0),y.size(0)
    return Kxx.sum()/(m*(m-1)) + Kyy.sum()/(n*(n-1)) - 2*Kxy.sum()/(m*n)


# ══════════════════════════════════════════════════════════════
# 2. WAE训练（支持缓存）
# ══════════════════════════════════════════════════════════════
def train_wae(comp_np, latent_dim=7, num_epoch=2300, cache_dir=None):
    """训练WAE，权重缓存到cache_dir，后续直接加载跳过训练。"""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f'wae_dim{latent_dim}.pth')
        if os.path.exists(cache_path):
            print(f"  [cache] Loading WAE dim={latent_dim} from {cache_path}")
            model = WAE(16, latent_dim)  # 原始BN结构，用于加载权重
            model.load_state_dict(
                torch.load(cache_path, map_location='cpu', weights_only=True))
            _replace_bn(model.encoder); _replace_bn(model.decoder)
            return model.eval().to(device)

    print(f"  Training WAE dim={latent_dim} ({num_epoch} epochs) ...")
    model = WAE(16, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.FloatTensor(comp_np)),
                        batch_size=32, shuffle=True, drop_last=False)
    best_loss = float('inf'); best_state = None

    for epoch in range(num_epoch):
        model.train(); ep_total = 0.0; n_batch = 0
        for (x,) in loader:
            x = x.to(device)
            z_prior = torch.randn(x.size(0), latent_dim).to(device)
            x_recon, z = model(x)
            loss = (F.kl_div((x_recon+1e-8).log(), x+1e-8, reduction='batchmean')
                    + 1e-2 * compute_mmd(z, z_prior)
                    + 0.1  * torch.abs(x_recon.sum(dim=1)-1.0).mean()
                    + 0.1  * F.relu(-x_recon).mean())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_total += loss.item(); n_batch += 1
        ep_total /= n_batch
        if ep_total < best_loss:
            best_loss = ep_total
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch+1) % 500 == 0:
            print(f"    epoch {epoch+1}/{num_epoch}  loss={ep_total:.6f}")

    model.load_state_dict(best_state)
    if cache_dir:
        torch.save(model.state_dict(), cache_path)  # 保存原始BN权重
        print(f"  Saved → {cache_path}")
    _replace_bn(model.encoder); _replace_bn(model.decoder)  # 推理时再替换
    return model.eval().to(device)


# ══════════════════════════════════════════════════════════════
# 3. WAE+MCMC生成（与generation_quality一致，加sigma/mg参数）
# ══════════════════════════════════════════════════════════════
def generate_wae_mcmc(wae, train_comp, n=N_GEN,
                      sigma_list=None, mg_lo=0.90, mg_hi=0.99,
                      n_chains=15, steps_per_chain=600):
    if sigma_list is None:
        sigma_list = [0.01, 0.02, 0.03, 0.04]
    latent_dim = wae.latent_dim

    wae.eval()
    with torch.no_grad():
        z_all_np = wae.encoder(
            torch.FloatTensor(train_comp).to(device)).cpu().numpy()

    valid_indices = np.where(
        [is_valid(c, mg_lo, mg_hi) for c in train_comp])[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(len(train_comp))

    N_START = min(n_chains, len(valid_indices))
    km = KMeans(n_clusters=N_START, random_state=42, n_init=10)
    km.fit(train_comp[valid_indices])
    start_indices = []
    for ci in range(N_START):
        members = np.where(km.labels_ == ci)[0]
        center  = km.cluster_centers_[ci]
        dists   = np.linalg.norm(
            train_comp[valid_indices[members]] - center, axis=1)
        start_indices.append(valid_indices[members[np.argmin(dists)]])
    start_indices = np.array(start_indices)

    samples_per_chain = max(1, (n + N_START - 1) // N_START)
    all_samples = []

    for vi in start_indices:
        cur_c = normalize_comp(train_comp[vi].copy())
        cur_z = z_all_np[vi].copy()
        cur_s = score_comp(cur_c, mg_lo, mg_hi)
        chain_samples = []; T = 2.0

        for _ in range(steps_per_chain):
            r = np.random.rand()
            if r < 0.50:
                sigma = np.random.choice(sigma_list)
                pc = normalize_comp(cur_c + np.random.randn(len(ELEMENTS))*sigma)
                if not is_valid(pc, mg_lo, mg_hi):
                    pc = post_process_comp(pc, mg_lo, mg_hi)
                with torch.no_grad():
                    pz = wae.encoder(torch.FloatTensor(pc).unsqueeze(0).to(device)
                                     ).cpu().numpy().flatten()
            elif r < 0.85:
                pz = cur_z + np.random.randn(latent_dim) * 0.35
                with torch.no_grad():
                    pc = wae.decoder(torch.FloatTensor(pz).unsqueeze(0).to(device)
                                     )[0].cpu().numpy().flatten()
                pc = post_process_comp(pc, mg_lo, mg_hi)
            else:
                idx = np.random.choice(valid_indices)
                pc  = normalize_comp(train_comp[idx] + np.random.randn(len(ELEMENTS))*0.01)
                if not is_valid(pc, mg_lo, mg_hi):
                    pc = post_process_comp(pc, mg_lo, mg_hi)
                with torch.no_grad():
                    pz = wae.encoder(torch.FloatTensor(pc).unsqueeze(0).to(device)
                                     ).cpu().numpy().flatten()

            ps = score_comp(pc, mg_lo, mg_hi)
            if np.random.rand() < min(1.0, (ps/(cur_s+1e-10))**(1.0/T)):
                cur_c, cur_z, cur_s = pc, pz, ps
            chain_samples.append(
                normalize_comp(post_process_comp(cur_c.copy(), mg_lo, mg_hi)))

        step_size = max(1, len(chain_samples)//samples_per_chain)
        all_samples.extend(
            chain_samples[step_size//2::step_size][:samples_per_chain])

    return np.array(all_samples[:n])


# ══════════════════════════════════════════════════════════════
# 4. 评估指标
# ══════════════════════════════════════════════════════════════
def compute_metrics(cands, train_comp, n_clusters=15):
    from sklearn.metrics import pairwise_distances_chunked

    def to_feat(comps):
        wt = np.array([atomic_to_wt(c) for c in comps]) * 100
        return wt[:, 1:]

    c_feat  = to_feat(cands)
    tr_feat = to_feat(train_comp)

    gen_pw   = pairwise_distances(c_feat).sum()  / (len(c_feat) *(len(c_feat) -1)+1e-10)
    train_pw = pairwise_distances(tr_feat).sum() / (len(tr_feat)*(len(tr_feat)-1)+1e-10)
    diversity = gen_pw / (train_pw + 1e-10)

    nn_dists, tr_nn = [], []
    for chunk in pairwise_distances_chunked(c_feat,  tr_feat, n_jobs=1):
        nn_dists.extend(chunk.min(axis=1).tolist())
    for chunk in pairwise_distances_chunked(tr_feat, tr_feat, n_jobs=1):
        d = chunk.copy(); np.fill_diagonal(d, np.inf)
        tr_nn.extend(d.min(axis=1).tolist())
    novelty = np.mean(nn_dists) / (np.mean(tr_nn) + 1e-10)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(tr_feat)
    coverage = len(set(km.predict(c_feat))) / n_clusters

    return dict(diversity=diversity, novelty=novelty, coverage=coverage)


# ══════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════
def run_sensitivity(n_gen=N_GEN, out_dir='sensitivity_results'):
    os.makedirs(out_dir, exist_ok=True)
    # 固定随机种子，保证图和Source Data来自同一次可复现的运行
    np.random.seed(42)
    torch.manual_seed(42)
    cache_dir = os.path.join(out_dir, 'wae_cache')

    print("Loading data ...")
    df = load_data(DATA_PATH)
    comp = df[ELEMENTS].values.astype(float)
    train_comp = comp / (comp.sum(axis=1, keepdims=True) + 1e-10)
    print(f"  {len(train_comp)} training compositions")

    all_results = {}

    # ── 参数1：latent_dim ─────────────────────────────────────
    # dim=7权重先从原始文件复制到cache，避免重新训练
    import shutil
    os.makedirs(cache_dir, exist_ok=True)
    cache7 = os.path.join(cache_dir, 'wae_dim7.pth')
    if not os.path.exists(cache7):
        # 兼容两种文件名（WAE_model_weight_pw.pth / WAE_model_weights_pw.pth）
        candidates = [
            WAE_WEIGHTS,
            os.path.join(SCRIPT_DIR, 'WAE_model_weights_pw.pth'),
            os.path.join(SCRIPT_DIR, 'WAE_model_weight_pw.pth'),
        ]
        src = next((p for p in candidates if os.path.exists(p)), None)
        if src is None:
            raise FileNotFoundError(
                "找不到WAE权重文件，请确认以下路径之一存在：\n"
                + '\n'.join(f"  {p}" for p in candidates))
        print(f"\nCopying WAE weights ({os.path.basename(src)}) to cache ...")
        shutil.copy2(src, cache7)

    print("\n=== Param 1: latent_dim ===")
    for vi, dim in enumerate(PARAM_CONFIGS['latent_dim']['values']):
        print(f"\n  dim={dim}")
        wae = train_wae(train_comp, latent_dim=dim,
                        num_epoch=2300, cache_dir=cache_dir)
        cands = generate_wae_mcmc(wae, train_comp, n=n_gen)
        all_results[('latent_dim', vi)] = cands
        m = compute_metrics(cands, train_comp)
        print(f"    diversity={m['diversity']:.3f}  "
              f"novelty={m['novelty']:.3f}  coverage={m['coverage']:.3f}")

    # ── 参数2：mcmc_sigma（dim=7固定）────────────────────────
    print("\n=== Param 2: mcmc_sigma ===")
    wae7 = train_wae(train_comp, latent_dim=7,
                     num_epoch=2300, cache_dir=cache_dir)
    for vi, sigma_list in enumerate(PARAM_CONFIGS['mcmc_sigma']['values']):
        print(f"\n  sigma={sigma_list}")
        cands = generate_wae_mcmc(wae7, train_comp, n=n_gen,
                                  sigma_list=sigma_list)
        all_results[('mcmc_sigma', vi)] = cands
        m = compute_metrics(cands, train_comp)
        print(f"    diversity={m['diversity']:.3f}  "
              f"novelty={m['novelty']:.3f}  coverage={m['coverage']:.3f}")

    # ── 参数3：mg_constraint（dim=7固定）─────────────────────
    print("\n=== Param 3: mg_constraint ===")
    for vi, (mg_lo, mg_hi) in enumerate(PARAM_CONFIGS['mg_constraint']['values']):
        print(f"\n  Mg=[{mg_lo:.0%}, {mg_hi:.0%}]")
        cands = generate_wae_mcmc(wae7, train_comp, n=n_gen,
                                  mg_lo=mg_lo, mg_hi=mg_hi)
        all_results[('mg_constraint', vi)] = cands
        m = compute_metrics(cands, train_comp)
        print(f"    diversity={m['diversity']:.3f}  "
              f"novelty={m['novelty']:.3f}  coverage={m['coverage']:.3f}")

    plot_sensitivity(all_results, train_comp, out_dir)
    print(f"\nDone → {out_dir}/")


# ══════════════════════════════════════════════════════════════
# 6. 可视化
# ══════════════════════════════════════════════════════════════
def plot_sensitivity(all_results, train_comp, out_dir):
    from sklearn.manifold import TSNE
    from scipy.stats import gaussian_kde
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import os

    RC = {
        'font.family':       'Arial',
        'font.size':         7,
        'axes.linewidth':    0.8,
        'axes.edgecolor':    'black',
        'axes.labelcolor':   'black',
        'xtick.color':       'black',
        'ytick.color':       'black',
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.labelsize':   6,
        'ytick.labelsize':   6,
        'text.color':        'black',
        'svg.fonttype':      'none',
        'pdf.fonttype':      42,
        'figure.facecolor':  'white',
        'axes.facecolor':    'white',
    }
    matplotlib.rcParams.update(RC)

    param_names = ['latent_dim', 'mcmc_sigma', 'mg_constraint']

    PARAM_CONFIGS = {
        'latent_dim': {
            'label':   'Latent space dimension',
            'current': 1,
            'tags':    ['dim = 5', 'dim = 7 (this work)', 'dim = 9'],
        },
        'mcmc_sigma': {
            'label':   'MCMC step size (σ)',
            'current': 1,
            'tags':    ['small [0.005–0.02]', 'medium [0.01–0.04] (this work)', 'large [0.02–0.08]'],
        },
        'mg_constraint': {
            'label':   'Mg content constraint',
            'current': 1,
            'tags':    ['loose [88–99%]', 'standard [90–99%] (this work)', 'strict [92–98%]'],
        },
    }

    def to_feat(comps):
        from generation_quality import atomic_to_wt
        wt = np.array([atomic_to_wt(c) for c in comps]) * 100
        return wt[:, 1:]

    # ── t-SNE ──
    print("\n  Computing t-SNE ...")
    train_feat = to_feat(train_comp)
    cands_list = list(all_results.values())
    n_per      = [len(c) for c in cands_list]

    all_feat = np.vstack([train_feat] + [to_feat(c) for c in cands_list])
    tsne = TSNE(n_components=2, perplexity=35, random_state=42,
                max_iter=1000, learning_rate='auto', init='pca')
    all_proj = tsne.fit_transform(all_feat)
    print("  t-SNE done")

    n_train    = len(train_feat)
    train_proj = all_proj[:n_train]
    offset = n_train
    cands_proj = []
    for npp in n_per:
        cands_proj.append(all_proj[offset:offset+npp])
        offset += npp

    # ── 保存Source Data：t-SNE坐标和成分（原子分数）──
    import pandas as pd
    rows = []
    for p, c in zip(train_proj, train_comp):
        rows.append({'group': 'Training set', 'tsne_1': p[0], 'tsne_2': p[1],
                     **{e: v for e, v in zip(ELEMENTS, c)}})
    for (pname, vi), proj, cands in zip(all_results.keys(), cands_proj, cands_list):
        grp = f"{PARAM_CONFIGS[pname]['label']}: {PARAM_CONFIGS[pname]['tags'][vi]}"
        for p, c in zip(proj, cands):
            rows.append({'group': grp, 'tsne_1': p[0], 'tsne_2': p[1],
                         **{e: v for e, v in zip(ELEMENTS, c)}})
    out_csv = os.path.join(out_dir, 'tsne_coordinates.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"  Source data saved → {out_csv}")

    pad = 2.0
    x0, x1 = all_proj[:,0].min()-pad, all_proj[:,0].max()+pad
    y0, y1 = all_proj[:,1].min()-pad, all_proj[:,1].max()+pad

    xg, yg   = np.mgrid[x0:x1:80j, y0:y1:80j]
    grid_pts  = np.vstack([xg.ravel(), yg.ravel()])

    train_kde    = gaussian_kde(train_proj.T, bw_method=0.25)
    train_z      = train_kde(grid_pts).reshape(xg.shape)
    train_thresh = np.percentile(train_z, 15)

    all_z_list = []
    for proj in cands_proj:
        try:
            z = gaussian_kde(proj.T, bw_method=0.25)(grid_pts).reshape(xg.shape)
        except Exception:
            z = np.zeros_like(xg)
        all_z_list.append(z)

    vmax_vals = []
    for z in all_z_list:
        z_m = z[train_z >= train_thresh]
        if len(z_m) > 0:
            vmax_vals.append(np.percentile(z_m, 98))
    global_vmax = float(np.median(vmax_vals)) if vmax_vals else 1.0

    # 统一刻度（取整数，3个刻度）
    x_ticks = np.linspace(np.ceil(x0), np.floor(x1), 3).astype(int)
    y_ticks = np.linspace(np.ceil(y0), np.floor(y1), 3).astype(int)

    # ── 画图 ──
    fig_w = 18.3 / 2.54
    fig, axes = plt.subplots(
        3, 3,
        figsize=(fig_w, fig_w * 1.05),
        gridspec_kw={
            'wspace': 0.35, 'hspace': 0.45,
            'left': 0.12, 'right': 0.92,
            'top':  0.93, 'bottom': 0.06,
        }
    )

    panel_letters = 'abcdefghi'
    idx = 0
    im_ref = None

    for ri, pname in enumerate(param_names):
        cfg = PARAM_CONFIGS[pname]
        for ci in range(3):
            ax = axes[ri, ci]
            z  = all_z_list[idx]

            im = ax.pcolormesh(xg, yg, z,
                               cmap='YlOrRd', shading='gouraud',
                               vmin=0, vmax=global_vmax, zorder=2)
            im_ref = im

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)

            # 坐标刻度
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.tick_params(axis='both', direction='in',
                           labelsize=6, colors='black',
                           top=True, right=True)

            # 轴标签每个panel都写
            ax.set_xlabel('t-SNE 1', fontsize=7, color='black',
                          labelpad=2, fontfamily='Arial')
            ax.set_ylabel('t-SNE 2', fontsize=7, color='black',
                          labelpad=2, fontfamily='Arial')

            # 四边封闭黑色
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.8)
                sp.set_color('black')

            # 列标题（第一行）
            if ri == 0:
                ax.set_title(
                    cfg['tags'][ci],
                    fontsize=7, pad=4,
                    color='black', fontfamily='Arial',
                )

            # panel letter
            ax.text(
                -0.18, 1.08,
                panel_letters[ri * 3 + ci],
                transform=ax.transAxes,
                fontsize=8, fontweight='bold',
                va='top', ha='left',
                color='black', fontfamily='Arial',
            )

            idx += 1

    # 行标签（每行最左侧panel的ylabel已有，再加一个居中行文字）
    row_labels = [cfg['label'] for cfg in
                  [PARAM_CONFIGS[p] for p in param_names]]
    for ri, label in enumerate(row_labels):
        fig.text(
            0.01,
            axes[ri, 0].get_position().y0 +
            (axes[ri, 0].get_position().height / 2),
            label,
            va='center', ha='left',
            rotation=90, fontsize=7,
            color='black', fontfamily='Arial',
        )
        # 清掉每个panel自带的ylabel避免重复
        for ci in range(3):
            axes[ri, ci].set_ylabel(
                't-SNE 2', fontsize=7, color='black',
                labelpad=2, fontfamily='Arial'
            )

    # colorbar
    cbar_ax = fig.add_axes([0.945, 0.20, 0.012, 0.28])
    cb = fig.colorbar(im_ref, cax=cbar_ax)
    cb.set_ticks([0, global_vmax])
    cb.set_ticklabels(['low', 'high'], fontsize=6, fontfamily='Arial')
    cb.ax.set_ylabel('Sampling density', fontsize=6,
                     labelpad=3, fontfamily='Arial', color='black')
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor('black')
    cb.ax.tick_params(labelcolor='black', width=0.6)

    for ext in ('png', 'svg'):
        fig.savefig(
            os.path.join(out_dir, f'sensitivity_kde.{ext}'),
            dpi=300, bbox_inches='tight', facecolor='white'
        )
    plt.close(fig)
    print("  sensitivity_kde (NC style) saved")


# ══════════════════════════════════════════════════════════════
# 7. 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default=DATA_PATH)
    parser.add_argument('--wae',   default=WAE_WEIGHTS)
    parser.add_argument('--n_gen', type=int, default=300)
    parser.add_argument('--out',   default='sensitivity_results')
    args = parser.parse_args()

    DATA_PATH   = args.data
    WAE_WEIGHTS = args.wae
    N_GEN       = args.n_gen

    run_sensitivity(n_gen=args.n_gen, out_dir=args.out)
    print("\nDone.")