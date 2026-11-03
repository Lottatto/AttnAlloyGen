"""
Feature Perturbation Analysis - NC Style
"""

import os
import pickle
import re
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore')

CUTOFF      = 2025
RESULTS_DIR = r'D:\vscode\corrosion rate\revision\K-fold\year_split_results_2025'
DATA_PATH   = 'database_colab_text_new1.xlsx'
OUTPUT_DIR  = r'D:\vscode\corrosion rate\revision\Interpretability'
SHAP_CSV    = os.path.join(OUTPUT_DIR, 'shap_importance.csv')

ELEMENT_COLS = {
    'Mg', 'Al', 'Zn', 'Mn', 'Zr', 'Y', 'Nd', 'Gd', 'Ca', 'Li',
    'Sc', 'Ce', 'La', 'Sn', 'In', 'Fe', 'Sr', 'Ag', 'Si', 'Cu',
    'Ni', 'Co', 'Bi', 'Sb', 'Pb', 'Be', 'Ba', 'Na', 'K', 'Pr',
    'Sm', 'Eu', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}

CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'

EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0',
    '10.1007/s11665-026-13708-2',
    '10.1016/j.jma.2018.11.004',
    '10.1016/j.jmrt.2022.05.161',
}

DOI_YEAR_MAP = {
    '10.1002/maco.201609141': 2016, '10.1002/sia.7075': 2022,
    '10.1007/s11661-021-06548-3': 2021, '10.1007/s11665-020-05034-y': 2020,
    '10.1007/s11665-022-07081-z': 2022, '10.1007/s11665-026-13543-5': 2026,
    '10.1007/s40195-022-01451-1': 2022, '10.1016/ j.jma.2022.12.005': 2022,
    '10.1016/S1005-0302(12)60186-8': 2012, '10.1016/j.apsusc.2016.01.283': 2016,
    '10.1016/j.apsusc.2020.147761': 2020, '10.1016/j.corcom.2022.10.002': 2022,
    '10.1016/j.corsci.2011.10.033': 2011, '10.1016/j.corsci.2017.01.022': 2017,
    '10.1016/j.corsci.2020.108451': 2020, '10.1016/j.corsci.2020.108487': 2020,
    '10.1016/j.corsci.2020.108831': 2020, '10.1016/j.corsci.2020.108958': 2020,
    '10.1016/j.corsci.2020.108998': 2020, '10.1016/j.corsci.2022.110861': 2022,
    '10.1016/j.corsci.2023.111021': 2023, '10.1016/j.corsci.2024.112180': 2024,
    '10.1016/j.electacta.2025.146957': 2025, '10.1016/j.jallcom.2020.153839': 2020,
    '10.1016/j.jallcom.2020.154344': 2020, '10.1016/j.jallcom.2020.156619': 2020,
    '10.1016/j.jallcom.2024.174068': 2024, '10.1016/j.jallcom.2025.180504': 2025,
    '10.1016/j.jallcom.2026.187172': 2026, '10.1016/j.jma.2016.12.002': 2016,
    '10.1016/j.jma.2018.09.002': 2018, '10.1016/j.jma.2020.03.014': 2020,
    '10.1016/j.jma.2020.06.004': 2020, '10.1016/j.jma.2022.08.003': 2022,
    '10.1016/j.jma.2026.102035': 2026, '10.1016/j.jmrt.2025.04.024': 2025,
    '10.1016/j.jmst.2020.05.067': 2020, '10.1016/j.jmst.2023.01.058': 2023,
    '10.1016/j.jmst.2023.08.068': 2023, '10.1016/j.jmst.2025.01.020': 2025,
    '10.1016/j.jpcs.2021.109952': 2021, '10.1016/j.matchar.2021.111294': 2021,
    '10.1016/j.matchar.2022.112121': 2022, '10.1016/j.matdes.2022.110980': 2022,
    '10.1016/j.mtcomm.2024.111277': 2024, '10.1016/j.scriptamat.2018.11.046': 2018,
    '10.1038/s41529-026-00755-2': 2026, '10.1088/2053-1591/ab51ab': 2019,
    '10.11900/0412.1961.2024.00230': 2024, '10.20964/2017.11.14': 2017,
}

FEATURE_DISPLAY_NAMES = {
    'Atomic weight mean':               'AWM',
    'Electronegativity local mismatch': 'ΔEN',
    'VEC mean':                         'VEC',
    'Average Ionization Energy':        'IE',
    'Average Work Function':            'WF',
    'Average IE Difference from Mg':    'IE/Mg',
    'Density':                          'ρ',
    'Average Melting Point':            'Tm',
    'Average Boiling Point':            'Tb',
    'Average Fermi Energy':             'EF',
    'Average Atomic Radius':            'R',
    'Average Surface Energy':           'γ',
    'Average Adsorption Energy':        'Eads',
}

# ─────────────────────────────────────────────
# 数据预处理
# ─────────────────────────────────────────────
def _standardize_immersion_time(text):
    text = str(text).strip().lower()
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)',   lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',    lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)', lambda m: f"{float(m.group(1)):.0f} min"),
    ]
    for pat, repl in patterns:
        m = re.search(pat, text)
        if m: return repl(m)
    return text

def _standardize_heat_treatment_process(text):
    text = str(text).strip().lower()
    mappings = {
        r'\bt4\b': 'solution-treated',
        r'\bt6\b': 'solution-treated+aging',
        r'\bt5\b': 'artificially aged',
        r'solution\s*treat(?:ed|ment)?': 'solution-treated',
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
        r'cast(?:ed|ing)?|as[\s-]?cast': 'cast',
        r'roll(?:ed|ing)?': 'rolling',
        r'extrud(?:ed|ing)|extrusion': 'extruded',
        r'forg(?:ed|ing)?': 'forging',
        r'no\s*deform|as[\s-]?received': 'no deformation',
    }
    for pat, repl in mappings.items():
        if re.search(pat, text, re.I): return repl
    return text

def apply_text_standardization(df):
    col_funcs = {
        'Immersion Time': _standardize_immersion_time,
        'Heat Treatment Process': _standardize_heat_treatment_process,
        'Heat Treatment Temperature': _standardize_heat_treatment_temperature,
        'Heat Treatment Time': _standardize_heat_treatment_time,
        'Deformation Process': _standardize_deformation_process,
    }
    for col, func in col_funcs.items():
        if col in df.columns:
            df[col] = df[col].apply(func)
    return df

def load_data():
    df = pd.read_excel(DATA_PATH)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try: df[col] = df[col].fillna(df[col].median())
            except: pass
    df = df[df[TARGET_COL] <= 30]
    mean_ = df[TARGET_COL].mean(); std_ = df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]
    df = df[df['Test method'] == 'pw'].copy()
    df = apply_text_standardization(df)
    num_cols_raw = df.columns[0:29].tolist()
    num_cols_raw = [c for c in num_cols_raw if df[c].dtype != 'object'
                    and c not in [TARGET_COL, DOI_COL]]
    zero_cols = [c for c in num_cols_raw if (df[c] == 0).all()]
    if zero_cols: df = df.drop(columns=zero_cols)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    df = df[df['year'].notna()].copy()
    return df

# ─────────────────────────────────────────────
# CombinedNet
# ─────────────────────────────────────────────
class CombinedNet(nn.Module):
    def __init__(self, numerical_size, categorical_info,
                 hidden_size=112, dropout_rate=0.12):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        for feature_name, num_categories in categorical_info.items():
            embedding_dim = (2 if num_categories <= 5
                             else 4 if num_categories <= 15 else 6)
            self.embeddings[feature_name] = nn.Embedding(num_categories, embedding_dim)
            total_embedding_dim += embedding_dim
        total_input_dim = numerical_size + total_embedding_dim
        self.fc1    = nn.Linear(total_input_dim, hidden_size)
        self.fc2    = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x_num, x_cat):
        embedded = [self.embeddings[fn](x_cat[:, i])
                    for i, fn in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embedded, dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.output(x)

def predict(model, scaler_y, X_num_scaled, X_cat_encoded):
    model.eval()
    with torch.no_grad():
        pred_s = model(torch.FloatTensor(X_num_scaled),
                       torch.LongTensor(X_cat_encoded)).numpy()
    return scaler_y.inverse_transform(pred_s).flatten()

# ─────────────────────────────────────────────
# 全局绘图参数（NC风格）
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       ['Arial', 'sans-serif'],
    'font.size':         11,
    'axes.labelsize':    11,
    'xtick.labelsize':   11,
    'ytick.labelsize':   11,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'in',
    'ytick.direction':   'in',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
})

def plot_results(single_df, cumulative_df, baseline_r2, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             gridspec_kw={'wspace': 0.45})

    # left: individual masking
    ax1 = axes[0]
    sd     = single_df.sort_values('delta_r2', ascending=True)
    labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in sd['feature']]
    delta  = sd['delta_r2'].values
    n_feat = len(labels)

    threshold = 0.05
    sig_mask  = -delta > threshold
    if sig_mask.any():
        sig_indices = np.where(sig_mask)[0]
        y_min = float(sig_indices.min()) - 0.5
        y_max = float(sig_indices.max()) + 0.5
        ax1.axhspan(y_min, y_max, color='#FFF3CD', alpha=0.6, zorder=0)
        x_max_val = float((-delta).max())
        x_arrow   = x_max_val * 0.50
        y_arrow   = (y_min + y_max) / 2
        ax1.annotate('Strong feature\ndependence',
                     xy=(x_max_val * 0.95, y_arrow),
                     xytext=(x_arrow, y_arrow),
                     fontsize=9, color='#B8860B',
                     arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1.2),
                     va='center', ha='left')

    colors = ['#6AB4DC'] * n_feat
    ax1.barh(range(n_feat), -delta, color=colors, height=0.6, zorder=2)
    ax1.set_yticks(range(n_feat))
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel('$\\Delta R^2$ drop when masked', fontsize=11, labelpad=4)
    ax1.axvline(0, color='#666666', linewidth=0.6, zorder=3)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)
    for sp in ax1.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    # right: cumulative masking
    ax2 = axes[1]
    y_min_data = float(cumulative_df['r2'].min())
    ax2.axhspan(y_min_data * 0.97, baseline_r2,
                color='#FDE8C8', alpha=0.5, zorder=0)
    ax2.axhline(baseline_r2, color='#888888', linewidth=0.8, linestyle='--',
                label=f'Baseline $R^2$={baseline_r2:.3f}', zorder=2)
    ax2.plot(cumulative_df['n_masked'], cumulative_df['r2'],
             'o-', color='#6AB4DC', linewidth=1.5, markersize=5,
             markerfacecolor='#6AB4DC', markeredgewidth=0.5,
             markeredgecolor='white', zorder=3)

    min_idx = cumulative_df['r2'].idxmin()
    x_low   = float(cumulative_df.loc[min_idx, 'n_masked'])
    y_low   = float(cumulative_df.loc[min_idx, 'r2'])
    ax2.annotate('$R^2$ drop',
                 xy=(x_low, y_low),
                 xytext=(x_low + 0.5, y_low + 0.05),
                 fontsize=9, color='#B85C00',
                 arrowprops=dict(arrowstyle='->', color='#B85C00', lw=1.2),
                 va='bottom', ha='left')

    ax2.set_xlabel('Masked features (cumulative,\ntop SHAP-ranked)', fontsize=11, labelpad=4)
    ax2.set_ylabel('Test $R^2$', fontsize=11, labelpad=4)
    ax2.set_xticks(cumulative_df['n_masked'])
    ax2.tick_params(axis='x', labelsize=11)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.legend(fontsize=10, frameon=False)
    for sp in ax2.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    plt.tight_layout()

    out_svg = os.path.join(output_dir, 'shap_perturbation.svg')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    print(f"Saved -> {out_svg}")
    plt.close()


def plot_mean_abs_delta(single_df, output_dir):
    sd     = single_df.sort_values('mean_abs_delta', ascending=True)
    labels = [FEATURE_DISPLAY_NAMES.get(f, f) for f in sd['feature']]
    values = sd['mean_abs_delta'].values
    n_feat = len(labels)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.barh(range(n_feat), values, color='#6AB4DC', height=0.6, zorder=2)

    for i, val in enumerate(values):
        ax.text(val + values.max() * 0.02, i, f'{val:.3f}',
                va='center', ha='left', fontsize=10)

    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Mean |$\\Delta$ predicted corrosion rate| (mm yr$^{-1}$)',
                  fontsize=11, labelpad=4)
    ax.set_xlim(0, values.max() * 1.3)
    ax.axvline(0, color='#666666', linewidth=0.6, zorder=3)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    plt.tight_layout()

    out_svg = os.path.join(output_dir, 'mean_abs_delta.svg')
    out_png = os.path.join(output_dir, 'mean_abs_delta.png')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f"Saved -> {out_png}")
    plt.close()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    print("Loading data...")
    df    = load_data()
    tr_df = df[df['year'] <  CUTOFF].copy()
    te_df = df[df['year'] >= CUTOFF].copy()
    print(f"Train: {len(tr_df)}  Test: {len(te_df)}")

    print("Loading model and preprocessors...")
    def _load(name):
        with open(os.path.join(RESULTS_DIR, f'{name}_cutoff{CUTOFF}.pkl'), 'rb') as f:
            return pickle.load(f)

    scaler_X         = _load('scaler_X')
    scaler_y         = _load('scaler_y')
    label_encoders   = _load('label_encoders')
    categorical_info = _load('categorical_info')
    num_cols         = _load('num_cols')

    model = CombinedNet(len(num_cols), categorical_info)
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, f'model_cutoff{CUTOFF}.pth'), map_location='cpu'))
    model.eval()

    num_cols_phys = [c for c in num_cols if c not in ELEMENT_COLS]
    print(f"Physical descriptors ({len(num_cols_phys)}): {num_cols_phys}")

    X_te_num = scaler_X.transform(te_df[num_cols].values.astype(float))
    X_te_cat = np.stack([
        label_encoders[col].transform(te_df[col].astype(str))
        for col in CATEGORICAL_COLUMNS], axis=1)
    y_te = te_df[TARGET_COL].values

    X_tr_s    = scaler_X.transform(tr_df[num_cols].values.astype(float))
    tr_mean_s = X_tr_s.mean(axis=0)

    y_pred_base = predict(model, scaler_y, X_te_num, X_te_cat)
    baseline_r2 = r2_score(y_te, y_pred_base)
    print(f"Baseline Test R² = {baseline_r2:.4f}")

    shap_df      = pd.read_csv(SHAP_CSV)
    top_features = [f for f in shap_df['Feature'].tolist() if f in num_cols_phys]
    print(f"SHAP-ranked features: {top_features}")

    single_rows = []
    for feat in top_features:
        col_i = num_cols.index(feat)
        X_p   = X_te_num.copy()
        X_p[:, col_i] = tr_mean_s[col_i]
        y_pred_masked  = predict(model, scaler_y, X_p, X_te_cat)
        r2_p           = r2_score(y_te, y_pred_masked)
        delta          = r2_p - baseline_r2
        mean_abs_delta = float(np.abs(y_pred_masked - y_pred_base).mean())
        single_rows.append({'feature': feat, 'r2': r2_p, 'delta_r2': delta,
                            'mean_abs_delta': mean_abs_delta})
        print(f"  Mask {FEATURE_DISPLAY_NAMES.get(feat,feat):12s}: "
              f"R²={r2_p:.4f}  ΔR²={delta:+.4f}  Mean|Δpred|={mean_abs_delta:.4f}")

    cumul_rows = []
    for n in range(1, min(len(top_features)+1, 9)):
        X_p = X_te_num.copy()
        for feat in top_features[:n]:
            col_i = num_cols.index(feat)
            X_p[:, col_i] = tr_mean_s[col_i]
        r2_p  = r2_score(y_te, predict(model, scaler_y, X_p, X_te_cat))
        delta = r2_p - baseline_r2
        cumul_rows.append({'n_masked': n, 'r2': r2_p, 'delta_r2': delta})
        print(f"  Cumulative top-{n}: R²={r2_p:.4f}  ΔR²={delta:+.4f}")

    single_df     = pd.DataFrame(single_rows)
    cumulative_df = pd.DataFrame(cumul_rows)
    single_df.to_csv(os.path.join(OUTPUT_DIR, 'perturbation_single.csv'), index=False)
    cumulative_df.to_csv(os.path.join(OUTPUT_DIR, 'perturbation_cumulative.csv'), index=False)

    plot_results(single_df, cumulative_df, baseline_r2, OUTPUT_DIR)
    plot_mean_abs_delta(single_df, OUTPUT_DIR)
    print("\nDone.")