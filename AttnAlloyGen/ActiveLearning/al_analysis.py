"""
al_analysis.py
==============
合并自 learning_curve.py + subset_analysis.py

输出：
  al_learning_curve.png / .svg   — 学习曲线（R² + RMSE，all + CR<10子集）
  al_scatter.png / .svg          — 散点图（5轮 × 2行）

只使用 cutoff=2025，只保留 all 和 CR<10 两个子集。

用法：python al_analysis.py
"""

import os, re, json, warnings, string
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ══════════════════════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════════════════════
BASE_DATA_PATH   = 'database_colab_text_new1.xlsx'
BASELINE_MODEL   = r'D:\vscode\corrosion rate\revision\generation\year_split_results_2025\model_cutoff2025.pth'
MODEL_BASE_DIR   = 'al_year_split_models'
BEST_PARAMS_PATH = 'al_best_params.json'
CUTOFF           = 2025

BASELINE_PARAMS = {'hidden_size': 112, 'dropout_rate': 0.12,
                   'learning_rate': 0.0006, 'seed': 119700}

# ══════════════════════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════════════════════
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]
NUMERICAL_START_COL = 0
NUMERICAL_END_COL   = 29

EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0', '10.1007/s11665-026-13708-2',
    '10.1016/j.jma.2018.11.004',  '10.1016/j.jmrt.2022.05.161',
}

DOI_YEAR_MAP = {
    '10.1002/maco.201609141': 2016, '10.1002/sia.7075': 2022,
    '10.1007/s11661-021-06548-3': 2021, '10.1007/s11665-020-05034-y': 2020,
    '10.1007/s11665-022-07081-z': 2022, '10.1007/s11665-023-08526-9': 2023,
    '10.1007/s11665-025-13079-0': 2025, '10.1007/s11665-026-13543-5': 2026,
    '10.1007/s11665-026-13708-2': 2026, '10.1007/s40195-022-01451-1': 2022,
    '10.1016/ j.jma.2022.12.005': 2022, '10.1016/S1005-0302(12)60186-8': 2012,
    '10.1016/j.apsusc.2016.01.283': 2016, '10.1016/j.apsusc.2020.147761': 2020,
    '10.1016/j.corcom.2022.10.002': 2022, '10.1016/j.corsci.2011.10.033': 2011,
    '10.1016/j.corsci.2017.01.022': 2017, '10.1016/j.corsci.2017.02.009': 2017,
    '10.1016/j.corsci.2020.108451': 2020, '10.1016/j.corsci.2020.108487': 2020,
    '10.1016/j.corsci.2020.108831': 2020, '10.1016/j.corsci.2020.108958': 2020,
    '10.1016/j.corsci.2020.108998': 2020, '10.1016/j.corsci.2021.110023': 2021,
    '10.1016/j.corsci.2022.110861': 2022, '10.1016/j.corsci.2023.111021': 2023,
    '10.1016/j.corsci.2024.112180': 2024, '10.1016/j.electacta.2025.146957': 2025,
    '10.1016/j.intermet.2023.108000': 2023, '10.1016/j.jallcom.2014.10.177': 2014,
    '10.1016/j.jallcom.2020.153839': 2020, '10.1016/j.jallcom.2020.154344': 2020,
    '10.1016/j.jallcom.2020.156619': 2020, '10.1016/j.jallcom.2024.174068': 2024,
    '10.1016/j.jallcom.2025.180504': 2025, '10.1016/j.jallcom.2026.187172': 2026,
    '10.1016/j.jma.2016.12.002': 2016, '10.1016/j.jma.2018.09.002': 2018,
    '10.1016/j.jma.2018.11.004': 2018, '10.1016/j.jma.2020.03.014': 2020,
    '10.1016/j.jma.2020.06.004': 2020, '10.1016/j.jma.2022.08.003': 2022,
    '10.1016/j.jma.2026.102035': 2026, '10.1016/j.jmrt.2022.05.161': 2022,
    '10.1016/j.jmrt.2025.04.024': 2025, '10.1016/j.jmst.2020.05.067': 2020,
    '10.1016/j.jmst.2023.01.058': 2023, '10.1016/j.jmst.2023.08.068': 2023,
    '10.1016/j.jmst.2025.01.020': 2025, '10.1016/j.jpcs.2021.109952': 2021,
    '10.1016/j.matchar.2021.111294': 2021, '10.1016/j.matchar.2022.112121': 2022,
    '10.1016/j.matdes.2022.110980': 2022, '10.1016/j.mtcomm.2024.111277': 2024,
    '10.1016/j.scriptamat.2018.11.046': 2018, '10.1038/s41529-026-00755-2': 2026,
    '10.1088/2053-1591/ab51ab': 2019, '10.11900/0412.1961.2024.00230': 2024,
    '10.20964/2017.11.14': 2017,
}

NPG_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488',
              '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']

# ══════════════════════════════════════════════════════════════════════════════
# 统一 rcParams（NC风格，与fig2a/2b/2c一致）
# ══════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    'font.family':       'Arial',
    'font.size':         7,
    'axes.linewidth':    0.75,
    'axes.labelsize':    7,
    'axes.titlesize':    7,
    'xtick.labelsize':   6.5,
    'ytick.labelsize':   6.5,
    'xtick.major.width': 0.75,
    'ytick.major.width': 0.75,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'legend.fontsize':   6,
    'legend.frameon':    False,
    'lines.linewidth':   1.0,
    'lines.markersize':  4,
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'svg.fonttype':      'none',
})

# ══════════════════════════════════════════════════════════════════════════════
# 工具：四边封闭box + 刻度朝内
# ══════════════════════════════════════════════════════════════════════════════
def _apply_box_style(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
        spine.set_color('black')
    ax.tick_params(axis='x', which='major',
                   top=True, bottom=True, direction='in',
                   width=0.75, length=3)
    ax.tick_params(axis='y', which='major',
                   left=True, right=True, direction='in',
                   width=0.75, length=3)
    ax.grid(False)

# ══════════════════════════════════════════════════════════════════════════════
# 文本标准化
# ══════════════════════════════════════════════════════════════════════════════
def _standardize_immersion_time(text):
    text = str(text).strip().lower()
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)',   lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',    lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)', lambda m: f"{float(m.group(1)):.0f} min"),
        (r'(\d+(?:\.\d+)?)\s*s(?:econds?)?(?:\s|$|,)', lambda m: f"{float(m.group(1)):.0f} s"),
    ]
    for pat, repl in patterns:
        m = re.search(pat, text)
        if m: return repl(m)
    return text

def _standardize_heat_treatment_process(text):
    text = str(text).strip().lower()
    mappings = {
        r'\bt4\b': 'solution-treated', r'\bt6\b': 'solution-treated+aging',
        r'\bt5\b': 'artificially aged',
        r'solution\s*treat(?:ed|ment)?(?:\s*\+\s*aging)?': 'solution-treated',
        r'anneal(?:ed|ing)?': 'annealed',
        r'stress[\s-]?relief': 'stress-relief treatment',
        r'aging|aged|age\s*treat': 'aging',
        r'no\s*heat\s*treat(?:ment)?|as[\s-]?cast|without': 'no heat treatment',
    }
    for pat, repl in mappings.items():
        if re.search(pat, text): return repl
    return text

def _standardize_heat_treatment_temperature(text):
    text = str(text).strip()
    if re.search(r'no|none|without|as.cast|n/a', text, re.I):
        return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', text)
    if m: return '+'.join([f"{float(t):.0f}°C" for t in m])
    return text

def _norm_time(text):
    text = str(text).strip().lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*h', text)
    if m: return f"{float(m.group(1)):.0f} hours"
    m = re.search(r'(\d+(?:\.\d+)?)\s*min', text)
    if m: return f"{float(m.group(1)):.0f} min"
    return text

def _standardize_heat_treatment_time(text):
    text = str(text).strip()
    if re.search(r'no|none|without|as.cast|n/a', text, re.I):
        return 'No heat treatment time'
    parts = re.split(r'[+,;/]', text)
    normed = [_norm_time(p) for p in parts if p.strip()]
    return '+'.join(normed) if normed else text

def _standardize_deformation_process(text):
    text = str(text).strip().lower()
    mappings = {
        r'cast(?:ed|ing)?|casted': 'cast', r'as[\s-]?cast': 'cast',
        r'roll(?:ed|ing)?': 'rolling', r'extrud(?:ed|ing)|extrusion': 'extruded',
        r'forg(?:ed|ing)?': 'forging', r'draw(?:n|ing)?': 'drawing',
        r'press(?:ed|ing)?|stamp': 'pressing', r'ECAP|equal\s*channel': 'ECAP',
        r'HPT|high\s*pressure\s*torsion': 'HPT', r'FSP|friction\s*stir': 'FSP',
        r'no\s*deform|as[\s-]?received': 'no deformation',
    }
    for pat, repl in mappings.items():
        if re.search(pat, text, re.I): return repl
    return text

def apply_text_standardization(df):
    col_funcs = {
        'Immersion Time':             _standardize_immersion_time,
        'Heat Treatment Process':     _standardize_heat_treatment_process,
        'Heat Treatment Temperature': _standardize_heat_treatment_temperature,
        'Heat Treatment Time':        _standardize_heat_treatment_time,
        'Deformation Process':        _standardize_deformation_process,
    }
    for col, func in col_funcs.items():
        if col in df.columns:
            df[col] = df[col].apply(func)
    return df

def fill_missing(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try:    df[col] = df[col].fillna(df[col].median())
            except: df[col] = df[col].fillna(df[col].mode()[0])
    return df

# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(path):
    df = pd.read_excel(path)
    df = fill_missing(df)
    df = df[df[TARGET_COL] <= 30]
    mean_, std_ = df[TARGET_COL].mean(), df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]
    df = df[df['Test method'] == 'pw'].copy()
    df = apply_text_standardization(df)
    num_cols = df.columns[NUMERICAL_START_COL:NUMERICAL_END_COL].tolist()
    num_cols = [c for c in num_cols if df[c].dtype != 'object'
                and c not in [TARGET_COL, DOI_COL]]
    zero_cols = [c for c in num_cols if (df[c] == 0).all()]
    if zero_cols: df = df.drop(columns=zero_cols)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    df = df[df['year'].notna()].copy()
    return df.reset_index(drop=True)

def load_iteration_data(path):
    df = pd.read_excel(path)
    df = fill_missing(df)
    df = apply_text_standardization(df)
    return df.drop(columns=[c for c in [DOI_COL, 'year'] if c in df.columns],
                   errors='ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════════════
class CombinedNet(nn.Module):
    def __init__(self, numerical_size, categorical_info,
                 hidden_size=112, dropout_rate=0.12):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb = 0
        for fn, nc in categorical_info.items():
            edim = 2 if nc <= 5 else (4 if nc <= 15 else 6)
            self.embeddings[fn] = nn.Embedding(nc, edim)
            total_emb += edim
        self.fc1    = nn.Linear(numerical_size + total_emb, hidden_size)
        self.fc2    = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x_num, x_cat):
        emb = [self.embeddings[fn](x_cat[:, i])
               for i, fn in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.dropout(x, self.dropout_rate, self.training)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.dropout(x, self.dropout_rate, self.training)
        return self.output(x)

# ══════════════════════════════════════════════════════════════════════════════
# 编码 & 推断
# ══════════════════════════════════════════════════════════════════════════════
def encode_split(tr_df, te_df, full_df, device):
    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in full_df.columns
                if full_df[c].dtype != 'object' and c not in exclude]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_tr_s = scaler_X.fit_transform(tr_df[num_cols].values.astype(float))
    X_te_s = scaler_X.transform(te_df[num_cols].values.astype(float))
    y_tr_s = scaler_y.fit_transform(tr_df[TARGET_COL].values.reshape(-1,1)).flatten()
    y_te_s = scaler_y.transform(te_df[TARGET_COL].values.reshape(-1,1)).flatten()

    cat_info, les = {}, {}
    cat_tr, cat_te = [], []
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(full_df[col].astype(str))
        cat_info[col] = len(le.classes_)
        les[col] = le
        cat_tr.append(le.transform(tr_df[col].astype(str)))
        cat_te.append(le.transform(te_df[col].astype(str)))

    return {
        'X_te_t':     torch.FloatTensor(X_te_s).to(device),
        'X_cat_te_t': torch.LongTensor(np.column_stack(cat_te)).to(device),
        'y_te_orig':  te_df[TARGET_COL].values,
        'year_te':    te_df['year'].values if 'year' in te_df.columns else np.zeros(len(te_df)),
        'scaler_y':   scaler_y,
        'cat_info':   cat_info,
        'num_cols':   num_cols,
        'input_size': X_tr_s.shape[1],
    }

def predict(model_path, enc, params, device):
    model = CombinedNet(enc['input_size'], enc['cat_info'],
                        int(params['hidden_size']),
                        float(params['dropout_rate'])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        pred_s = model(enc['X_te_t'], enc['X_cat_te_t']).cpu().numpy()
    return enc['scaler_y'].inverse_transform(pred_s).flatten()



def predict_mc_dropout(model_path, enc, params, device, n_passes=30):
    """MC Dropout预测：保持dropout激活，多次前向传播取均值和标准差。"""
    model = CombinedNet(enc['input_size'], enc['cat_info'],
                        int(params['hidden_size']),
                        float(params['dropout_rate'])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.train()  # 保持dropout激活
    preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            pred_s = model(enc['X_te_t'], enc['X_cat_te_t']).cpu().numpy()
            preds.append(enc['scaler_y'].inverse_transform(pred_s).flatten())
    preds = np.array(preds)  # (n_passes, n_samples)
    mu  = np.maximum(preds.mean(axis=0), 0.0)
    sig = preds.std(axis=0)
    return mu, sig
# ══════════════════════════════════════════════════════════════════════════════
# 指标计算
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(true, pred, years=None, threshold=None):
    if threshold is not None:
        mask = true < threshold
        if mask.sum() < 3:
            return {'r2': np.nan, 'rmse': np.nan, 'n': mask.sum(),
                    'true': true[mask], 'pred': pred[mask],
                    'years': years[mask] if years is not None else None}
        true, pred = true[mask], pred[mask]
        if years is not None: years = years[mask]
    return {
        'r2':   r2_score(true, pred),
        'rmse': np.sqrt(mean_squared_error(true, pred)),
        'n':    len(true),
        'true': true,
        'pred': pred,
        'years': years,
    }

# ══════════════════════════════════════════════════════════════════════════════
# 图(g)：学习曲线
# ══════════════════════════════════════════════════════════════════════════════
def plot_learning_curve(records):
    C_GRAY = '#555555'
    C_BLUE = '#6AB4DC'

    rmse_all  = [r['rmse_all']  for r in records]
    rmse_lt10 = [r['rmse_lt10'] for r in records]
    r2_all    = [r['r2_all']    for r in records]
    r2_lt10   = [r['r2_lt10']   for r in records]
    x         = list(range(len(records)))
    x_labels  = [r['name'] for r in records]

    fig, (ax_r2, ax_rmse) = plt.subplots(2, 1, figsize=(3.5, 4.2), sharex=True)
    fig.subplots_adjust(hspace=0.10)

    # R²
    ax_r2.plot(x, r2_all, 'o-', color=C_GRAY,
               markerfacecolor=C_GRAY, markeredgewidth=0,
               label='All test samples', zorder=3)
    ax_r2.plot(x, r2_lt10, 's-', color=C_BLUE,
               markerfacecolor=C_BLUE, markeredgewidth=0,
               label='CR < 10 mm yr$^{-1}$', zorder=3)
    for xi, v in zip(x, r2_all):
        ax_r2.text(xi, v + 0.008, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=5.5, color=C_GRAY)
    for xi, v in zip(x, r2_lt10):
        if not np.isnan(v):
            ax_r2.text(xi, v - 0.012, f'{v:.3f}',
                       ha='center', va='top', fontsize=5.5, color=C_BLUE)
    ax_r2.set_ylabel('Test $R^2$', fontsize=7, labelpad=4)
    ax_r2.legend(loc='lower right', fontsize=6,
                 handletextpad=0.4, labelspacing=0.3)
    _apply_box_style(ax_r2)

    # RMSE
    ax_rmse.plot(x, rmse_all, 'o-', color=C_GRAY,
                 markerfacecolor=C_GRAY, markeredgewidth=0, zorder=3)
    ax_rmse.plot(x, rmse_lt10, 's-', color=C_BLUE,
                 markerfacecolor=C_BLUE, markeredgewidth=0, zorder=3)
    for xi, v in zip(x, rmse_all):
        ax_rmse.text(xi, v + 0.008, f'{v:.3f}',
                     ha='center', va='bottom', fontsize=5.5, color=C_GRAY)
    for xi, v in zip(x, rmse_lt10):
        if not np.isnan(v):
            ax_rmse.text(xi, v - 0.012, f'{v:.3f}',
                         ha='center', va='top', fontsize=5.5, color=C_BLUE)
    ax_rmse.set_ylabel('RMSE (mm yr$^{-1}$)', fontsize=7, labelpad=4)
    ax_rmse.set_xlabel('Active learning iteration', fontsize=7, labelpad=4)
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(x_labels, fontsize=6.5, rotation=15, ha='right')
    _apply_box_style(ax_rmse)

    plt.tight_layout(pad=0.8)
    for fmt in ('png', 'svg'):
        plt.savefig(f'al_learning_curve.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("学习曲线: al_learning_curve.png / .svg")

# ══════════════════════════════════════════════════════════════════════════════
# 图(d)：散点图
# ══════════════════════════════════════════════════════════════════════════════
def plot_scatter(scatter_data):
    if not scatter_data:
        print("[警告] scatter_data 为空，跳过散点图")
        return

    iter_order = ['Baseline', '1st Iteration', '2nd Iteration',
                  '3rd Iteration', '4th Iteration']
    row_keys   = ['all', 'lt10']
    row_titles = ['All test samples', 'CR < 10 mm yr$^{-1}$']
    panel_labels = list(string.ascii_lowercase[:10])

    all_years = set()
    for it in iter_order:
        if it in scatter_data:
            for rk in row_keys:
                m = scatter_data[it][rk]
                if m.get('years') is not None:
                    all_years.update(m['years'])
    unique_years = sorted(all_years)
    year_color   = {yr: NPG_COLORS[i % len(NPG_COLORS)]
                    for i, yr in enumerate(unique_years)}

    n_cols = len(iter_order)
    n_rows = 2
    sub_w  = 3.5 / 2.54
    sub_h  = 3.5 / 2.54
    fig_w  = sub_w * n_cols + 0.8
    fig_h  = sub_h * n_rows + 0.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={'wspace': 0.45, 'hspace': 0.55})
    fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.11)

    panel_idx = 0
    for row_i, (row_key, row_title) in enumerate(zip(row_keys, row_titles)):
        for col_i, iter_name in enumerate(iter_order):
            ax = axes[row_i][col_i]
            label = panel_labels[panel_idx]; panel_idx += 1

            if iter_name not in scatter_data:
                ax.axis('off'); continue

            m = scatter_data[iter_name][row_key]
            if np.isnan(m.get('r2', np.nan)) or m['n'] < 3:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=7)
                ax.axis('off'); continue

            true_v  = m['true']
            pred_v  = m['pred']
            years_v = m['years'] if m.get('years') is not None \
                      else np.zeros(len(true_v))

            vmax = float(np.ceil(max(true_v.max(), pred_v.max()) * 1.05))
            vmax = max(vmax, 1.0)

            sigma_v = m.get('sigma', np.zeros(len(true_v)))

            for yr in unique_years:
                mask = years_v == yr
                if mask.sum() == 0: continue
                ax.errorbar(pred_v[mask], true_v[mask],
                            xerr=sigma_v[mask],
                            fmt='o', color=year_color[yr],
                            markersize=3.0,
                            ecolor=year_color[yr], elinewidth=0.5,
                            capsize=1.5, capthick=0.5,
                            alpha=0.55, zorder=3, label=str(int(yr)))

            ax.plot([0, vmax], [0, vmax],
                    color='#888888', linewidth=0.8,
                    linestyle=(0, (4, 3)), zorder=2)

            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_aspect('equal', adjustable='box')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, integer=True))

            _apply_box_style(ax)

            if col_i == 0:
                ax.set_ylabel('Experimental CR\n(mm yr$^{-1}$)',
                              fontsize=7, labelpad=3)
            if row_i == n_rows - 1:
                ax.set_xlabel('Predicted CR\n(mm yr$^{-1}$)',
                              fontsize=7, labelpad=3)

            ax.text(0.05, 0.97,
                    f'$R^2$\u2009=\u2009{m["r2"]:.3f}\n'
                    f'RMSE\u2009=\u2009{m["rmse"]:.3f}',
                    transform=ax.transAxes, fontsize=6,
                    va='top', ha='left', linespacing=1.4)

            if row_i == 0:
                ax.set_title(iter_name, fontsize=7, pad=3)

            if col_i == n_cols - 1:
                ax.text(1.05, 0.5, row_title,
                        transform=ax.transAxes, fontsize=6.5,
                        va='center', ha='left', rotation=270)

            if row_i == 0 and col_i == 0:
                ax.legend(title='Year', fontsize=6,
                          title_fontsize=6, loc='lower right',
                          frameon=False, labelspacing=0.2,
                          handletextpad=0.3, markerscale=1.2)

            ax.text(-0.20, 1.18, label,
                    transform=ax.transAxes,
                    fontsize=8, fontweight='bold', va='top')

    for fmt in ('png', 'svg'):
        plt.savefig(f'al_scatter.{fmt}', dpi=300, bbox_inches='tight')
    plt.close()
    print("散点图: al_scatter.png / .svg")

# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    print("\n[加载数据]")
    base_df    = load_and_preprocess(BASE_DATA_PATH)
    train_base = base_df[base_df['year'] < CUTOFF].copy().reset_index(drop=True)
    test_fixed = base_df[base_df['year'] >= CUTOFF].copy().reset_index(drop=True)
    print(f"Train base: {len(train_base)} | Test: {len(test_fixed)}")
    print(f"Test CR<10: {(test_fixed[TARGET_COL]<10).sum()} 条")

    iteration_order = [
        ('Baseline',      None,                 BASELINE_MODEL, BASELINE_PARAMS),
        ('1st Iteration', '1st_iteration.xlsx', None,           None),
        ('2nd Iteration', '2nd_iteration.xlsx', None,           None),
        ('3rd Iteration', '3rd_iteration.xlsx', None,           None),
        ('4th Iteration', '4th_iteration.xlsx', None,           None),
    ]

    records     = []
    scatter_data = {}
    export_rows  = []   # 导出散点数据，用于 Source Data
    current_train = train_base.copy()

    for iter_name, iter_path, model_path_override, params_override in iteration_order:
        print(f"\n{'='*50}\n[{iter_name}]")

        if iter_path is not None:
            new_data = load_iteration_data(iter_path)
            current_train = pd.concat([current_train, new_data], ignore_index=True)
        n_train = len(current_train)
        print(f"  累积训练集: {n_train} 条")

        full_df = pd.concat([current_train, test_fixed], ignore_index=True)
        enc = encode_split(current_train, test_fixed, full_df, device)

        if model_path_override is not None:
            mp, params = model_path_override, params_override
        else:
            mp = os.path.join(MODEL_BASE_DIR, f'cutoff{CUTOFF}',
                              f"{iter_name.replace(' ','_').lower()}_model.pth")
            iter_key = f"{iter_name}_cutoff{CUTOFF}"
            params = best_params.get(iter_key)
            if params is None:
                print(f"  [警告] 找不到 {iter_key} 参数，跳过")
                continue

        if not os.path.exists(mp):
            print(f"  [警告] 找不到模型 {mp}，跳过")
            continue

        pred_te = predict(mp, enc, params, device)
        mu_te, sig_te = predict_mc_dropout(mp, enc, params, device, n_passes=30)
        true_te = enc['y_te_orig']
        year_te = enc['year_te']

        m_all  = compute_metrics(true_te, pred_te, years=year_te)
        m_all['sigma'] = sig_te
        m_lt10 = compute_metrics(true_te, pred_te, years=year_te, threshold=10)
        # lt10子集的sigma：只保留CR<10的点对应的sigma
        lt10_mask = true_te < 10
        m_lt10['sigma'] = sig_te[lt10_mask]
        for i in range(len(true_te)):
            export_rows.append({
                'Iteration': iter_name,
                'Year': int(year_te[i]),
                'Experimental CR (mm/yr)': true_te[i],
                'Predicted CR (mm/yr)': pred_te[i],
                'MC Dropout sigma (mm/yr)': sig_te[i],
                'CR < 10 subset': bool(true_te[i] < 10),
            })

        print(f"  all  → R²={m_all['r2']:.4f}, RMSE={m_all['rmse']:.4f}")
        print(f"  <10  → R²={m_lt10['r2']:.4f}, RMSE={m_lt10['rmse']:.4f}, n={m_lt10['n']}")

        records.append({
            'name':      iter_name,
            'n_train':   n_train,
            'rmse_all':  m_all['rmse'],
            'rmse_lt10': m_lt10['rmse'],
            'r2_all':    m_all['r2'],
            'r2_lt10':   m_lt10['r2'],
        })
        scatter_data[iter_name] = {'all': m_all, 'lt10': m_lt10}

    print(f"\n{'='*60}")
    print(f"{'Model':<18} {'N':>5} {'R²(all)':>9} {'RMSE(all)':>10} {'RMSE(<10)':>10}")
    print(f"{'-'*55}")
    for r in records:
        print(f"{r['name']:<18} {r['n_train']:>5} {r['r2_all']:>9.4f} "
              f"{r['rmse_all']:>10.4f} {r['rmse_lt10']:>10.4f}")

    pd.DataFrame(export_rows).to_csv('al_scatter_data.csv', index=False,
                                     encoding='utf-8-sig')
    print("已导出: al_scatter_data.csv")

    plot_learning_curve(records)
    plot_scatter(scatter_data)
    print("\n✅ 完成！")


if __name__ == '__main__':
    main()