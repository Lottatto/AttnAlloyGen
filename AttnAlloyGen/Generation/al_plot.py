import os, sys, argparse, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from sklearn.manifold import TSNE

# ── 元素列表（和 al_ablation.py 保持一致）──────────────────
ELEMENTS = ['Mg','Al','Zn','Li','Y','Mn','Nd','Ce','Ca','Gd','Zr','Sc','Fe','Sn','In','La']

try:
    from pymatgen.core.periodic_table import Element
    _aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
    def atomic_to_wt(comp):
        w = comp * _aw
        return w / (w.sum() + 1e-10)
except ImportError:
    # fallback：近似原子量（不影响可视化相对关系）
    _AW = {'Mg':24.3,'Al':27.0,'Zn':65.4,'Li':6.9,'Y':88.9,'Mn':54.9,
           'Nd':144.2,'Ce':140.1,'Ca':40.1,'Gd':157.2,'Zr':91.2,
           'Sc':45.0,'Fe':55.8,'Sn':118.7,'In':114.8,'La':138.9}
    _aw = np.array([_AW[e] for e in ELEMENTS])
    def atomic_to_wt(comp):
        w = comp * _aw
        return w / (w.sum() + 1e-10)

# ══════════════════════════════════════════════════════════════
# 全局绘图参数
# ══════════════════════════════════════════════════════════════
def set_rcparams():
    matplotlib.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size':         7,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'pdf.fonttype':      42,
        'svg.fonttype':      'none',
    })


# ══════════════════════════════════════════════════════════════
# 图1：Best-so-far CR 主图
# ══════════════════════════════════════════════════════════════
def plot_best_cr(all_histories, out_dir):
    set_rcparams()

    max_it = max(
        h['iteration'][-1]
        for info in all_histories.values()
        for h in info['trials']
    )
    iters = np.arange(0, max_it + 1)

    def get_mat(info, key):
        return np.array([
            np.interp(iters, h['iteration'], h[key])
            for h in info['trials']
        ])

    fig, ax = plt.subplots(figsize=(8.8/2.54, 7.0/2.54))

    # 基线虚线
    baseline_cr = list(all_histories.values())[0]['trials'][0]['best_cr'][0]
    ax.axhline(y=baseline_cr, color='gray', ls='--', lw=0.8,
               alpha=0.7, label=f'Initial best ({baseline_cr:.3f})')

    markers = {'exploitation': 'o', 'ucb': 's', 'ei': '^', 'random': 'D'}

    for acq, info in all_histories.items():
        mat    = get_mat(info, 'best_cr')
        mu     = mat.mean(0)
        std    = mat.std(0)
        ax.errorbar(iters, mu, yerr=std,
                    fmt=info['ls'], color=info['color'], lw=1.2,
                    marker=markers.get(acq, 'o'), ms=4,
                    markerfacecolor='white', markeredgewidth=0.8,
                    elinewidth=0.8, capsize=2, capthick=0.8,
                    label=info['label'], zorder=3)

    ax.set_xlabel('Number of iterations', fontsize=7)
    ax.set_ylabel('Best-so-far corrosion rate (mm yr$^{-1}$)', fontsize=7)
    ax.set_xlim(-0.5, max_it + 0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc='lower left',
              handlelength=2.0, handletextpad=0.5)
    fig.tight_layout(pad=0.5)

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'al_best_cr.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  al_best_cr.png saved")


# ══════════════════════════════════════════════════════════════
# 图2：辅图（test R²、test RMSE、归一化 novelty）
# ══════════════════════════════════════════════════════════════
def plot_model_perf(all_histories, out_dir):
    set_rcparams()

    max_it = max(
        h['iteration'][-1]
        for info in all_histories.values()
        for h in info['trials']
    )
    iters = np.arange(0, max_it + 1)

    def get_mat(info, key):
        return np.array([
            np.interp(iters, h['iteration'], h[key])
            for h in info['trials']
        ])

    fig, axes = plt.subplots(1, 3, figsize=(18.0/2.54, 5.5/2.54),
                              gridspec_kw={'wspace': 0.40})
    ax_r2, ax_rmse, ax_div = axes

    for acq, info in all_histories.items():
        color = info['color']
        ls    = info['ls']
        label = info['label']

        for ax, key, ylabel in [
            (ax_r2,   'test_r2',   'Test R²'),
            (ax_rmse, 'test_rmse', 'Test RMSE (mm yr$^{-1}$)'),
        ]:
            mat = get_mat(info, key)
            mu  = mat.mean(0)
            std = mat.std(0)
            ax.plot(iters, mu, color=color, ls=ls, lw=1.2, label=label)
            ax.fill_between(iters, mu - std, mu + std,
                            color=color, alpha=0.10)

        div_mat = get_mat(info, 'diversity')
        mu_div  = div_mat.mean(0)[1:]
        std_div = div_mat.std(0)[1:]
        iters_div = iters[1:]
        ax_div.plot(iters_div, mu_div, color=color, ls=ls, lw=1.2, label=label)
        ax_div.fill_between(iters_div, mu_div - std_div, mu_div + std_div,
                             color=color, alpha=0.10)

    # 在 novelty 图上加 y=1 参考线（=训练集内部密度基准）
    ax_div.axhline(y=1.0, color='gray', ls=':', lw=0.7, alpha=0.6)

    for ax, ylabel, panel in [
        (ax_r2,   'Test R²',                                   'a'),
        (ax_rmse, 'Test RMSE (mm yr$^{-1}$)',                  'b'),
        (ax_div,  'Cumulative novelty\n(normalised NN dist)',   'c'),
    ]:
        ax.set_xlabel('Number of iterations', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlim(left=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=5.0, frameon=False, handlelength=1.5)
        ax.text(-0.18, 1.08, panel, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    # 强制关闭 novelty 轴的科学计数法
    ax_div.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax_div.yaxis.get_offset_text().set_visible(False)

    fig.tight_layout(pad=0.5)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'al_model_perf.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  al_model_perf.png saved")


# ══════════════════════════════════════════════════════════════
# 图3：t-SNE 成分空间选点分布
# ══════════════════════════════════════════════════════════════
def plot_tsne(all_histories, train_init_elements, out_dir):
    set_rcparams()

    # 初始训练集 wt%（非Mg，15维）
    init_comp = train_init_elements.astype(float)
    init_comp = init_comp / (init_comp.sum(axis=1, keepdims=True) + 1e-10)
    init_wt   = np.array([atomic_to_wt(c) for c in init_comp])[:, 1:] * 100

    strat_order = list(all_histories.keys())

    # 收集所有策略所有trial的选点
    all_sel_wt   = []
    all_sel_meta = []   # (strat_idx, iteration)
    all_sel_trial = []  # trial number, for Source Data

    for si, acq in enumerate(strat_order):
        info = all_histories[acq]
        for ti, h in enumerate(info['trials']):
            for it_idx, sel_wt in enumerate(h.get('selected_wt', [])):
                it = h['selected_it'][it_idx]
                for pt in sel_wt:
                    all_sel_wt.append(pt)
                    all_sel_meta.append((si, it))
                    all_sel_trial.append(ti + 1)

    all_sel_wt = np.array(all_sel_wt)

    # 联合 t-SNE
    combined = np.vstack([init_wt, all_sel_wt])
    n_init   = len(init_wt)

    print("  Fitting t-SNE...", flush=True)
    tsne = TSNE(n_components=2,
                perplexity=min(30, max(5, len(combined)//4)),
                random_state=42, max_iter=1000,
                learning_rate='auto', init='pca', n_jobs=1)
    emb      = tsne.fit_transform(combined)
    init_emb = emb[:n_init]
    sel_emb  = emb[n_init:]

    # ── Source Data：t-SNE坐标 + 成分（wt%，不含Mg）──
    import pandas as pd
    wt_cols = [f'{e} (wt.%)' for e in ELEMENTS[1:]]
    rows = []
    for p, w in zip(init_emb, init_wt):
        rows.append({'group': 'Initial training set', 'strategy': '', 'trial': '',
                     'iteration': '', 'tsne_1': p[0], 'tsne_2': p[1],
                     **dict(zip(wt_cols, w))})
    for p, w, (si, it), tr in zip(sel_emb, all_sel_wt, all_sel_meta, all_sel_trial):
        rows.append({'group': 'Selected', 'strategy': all_histories[strat_order[si]]['label'],
                     'trial': tr, 'iteration': it, 'tsne_1': p[0], 'tsne_2': p[1],
                     **dict(zip(wt_cols, w))})
    out_csv = os.path.join(out_dir, 'al_tsne_coordinates.csv')
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"  Source data saved -> {out_csv}")

    # 1行4列
    fig, axes = plt.subplots(1, 4, figsize=(18.0/2.54, 5.0/2.54),
                              gridspec_kw={'wspace': 0.06})

    max_it_global = max(m[1] for m in all_sel_meta) if all_sel_meta else 1

    for si, (acq, ax) in enumerate(zip(strat_order, axes)):
        info  = all_histories[acq]
        color = info['color']
        label = info['label']

        # 背景：初始训练集密度云
        ax.scatter(init_emb[:, 0], init_emb[:, 1],
                   s=8, c='#D0D0D0', alpha=0.55, linewidths=0,
                   rasterized=True, zorder=1)

        # 前景：本策略选点，透明度随迭代深度增加
        mask     = np.array([m[0] == si for m in all_sel_meta])
        iters_sel = np.array([m[1] for m in all_sel_meta])

        if mask.sum() > 0:
            pts = sel_emb[mask]
            its = iters_sel[mask]
            base_rgb = np.array(mcolors.to_rgb(color))

            for it_val in np.unique(its):
                m2  = its == it_val
                alp = float(0.20 + 0.70 * (it_val / max_it_global))
                rgba = (*base_rgb, alp)
                ax.scatter(pts[m2, 0], pts[m2, 1],
                           s=22, c=[rgba] * int(m2.sum()),
                           linewidths=0.4, edgecolors='white', zorder=3)

        # 标题
        ax.set_title(label, fontsize=6.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#BBBBBB')

    # 第一个子图加轴标签
    axes[0].set_xlabel('t-SNE 1', fontsize=6)
    axes[0].set_ylabel('t-SNE 2', fontsize=6)

    # 右侧颜色条（迭代轮次，浅→深）
    cbar_ax = fig.add_axes([0.92, 0.18, 0.013, 0.64])
    sm = plt.cm.ScalarMappable(
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            'iter_grey', ['#EEEEEE', '#333333']),
        norm=plt.Normalize(1, max_it_global))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Iteration', fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5)

    fig.subplots_adjust(right=0.90)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'al_tsne.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  al_tsne.png saved")


# ══════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='al_ablation_results/all_histories.pkl',
                        help='pickle文件路径')
    parser.add_argument('--out', default=None,
                        help='输出目录（默认和pkl同目录）')
    args = parser.parse_args()

    if not os.path.exists(args.pkl):
        print(f"ERROR: {args.pkl} not found. Run al_ablation.py first.")
        sys.exit(1)

    with open(args.pkl, 'rb') as f:
        data = pickle.load(f)

    all_histories       = data['all_histories']
    train_init_elements = data['train_init_elements']

    out_dir = args.out or os.path.dirname(args.pkl)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loaded: {args.pkl}")
    print(f"Output: {out_dir}")
    print("Generating figures...")

    plot_best_cr(all_histories, out_dir)
    plot_model_perf(all_histories, out_dir)
    plot_tsne(all_histories, train_init_elements, out_dir)

    print("Done.")