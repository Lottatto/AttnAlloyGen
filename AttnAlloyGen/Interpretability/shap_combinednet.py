"""
SHAP Analysis for CombinedNet — Numerical Features Only
========================================================
固定类别特征为训练集众数，只对数值特征做SHAP分析。
需要先运行 year_split_replay.py 生成以下文件：
  year_split_results_2025/model_cutoff2025.pth
  year_split_results_2025/scaler_X_cutoff2025.pkl
  year_split_results_2025/scaler_y_cutoff2025.pkl
  year_split_results_2025/label_encoders_cutoff2025.pkl
  year_split_results_2025/categorical_info_cutoff2025.pkl
  year_split_results_2025/num_cols_cutoff2025.pkl

输出到：D:/vscode/corrosion rate/revision/Interpretability/
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import RobustScaler, LabelEncoder

# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────
CUTOFF       = 2025
RESULTS_DIR  = r'D:\vscode\corrosion rate\revision\K-fold\year_split_results_2025'
DATA_PATH    = 'database_colab_text_new1.xlsx'
OUTPUT_DIR   = r'D:\vscode\corrosion rate\revision\Interpretability'

# 排除元素成分列，只保留物理描述符
ELEMENT_COLS = {
    'Mg', 'Al', 'Zn', 'Mn', 'Zr', 'Y', 'Nd', 'Gd', 'Ca', 'Li',
    'Sc', 'Ce', 'La', 'Sn', 'In', 'Fe', 'Sr', 'Ag', 'Si', 'Cu',
    'Ni', 'Co', 'Bi', 'Sb', 'Pb', 'Be', 'Ba', 'Na', 'K', 'Pr',
    'Sm', 'Eu', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
}
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 特征显示名称映射（全部使用缩写）
# ─────────────────────────────────────────────
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
# 数据加载（与replay保持完全一致）
# ─────────────────────────────────────────────
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
CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _standardize_immersion_time(text):
    text = str(text).strip().lower()
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)',  lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',   lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)',lambda m: f"{float(m.group(1)):.0f} min"),
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
        r'forg(?:ed|ing)?': 'forging', r'no\s*deform|as[\s-]?received': 'no deformation',
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
# CombinedNet（与replay完全一致）
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

# ─────────────────────────────────────────────
# 加载模型和预处理对象
# ─────────────────────────────────────────────
def load_model_and_preprocessors():
    with open(os.path.join(RESULTS_DIR, f'scaler_X_cutoff{CUTOFF}.pkl'), 'rb') as f:
        scaler_X = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, f'scaler_y_cutoff{CUTOFF}.pkl'), 'rb') as f:
        scaler_y = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, f'label_encoders_cutoff{CUTOFF}.pkl'), 'rb') as f:
        label_encoders = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, f'categorical_info_cutoff{CUTOFF}.pkl'), 'rb') as f:
        categorical_info = pickle.load(f)
    with open(os.path.join(RESULTS_DIR, f'num_cols_cutoff{CUTOFF}.pkl'), 'rb') as f:
        num_cols = pickle.load(f)

    params = {'hidden_size': 112, 'dropout_rate': 0.12}
    model = CombinedNet(len(num_cols), categorical_info,
                        hidden_size=params['hidden_size'],
                        dropout_rate=params['dropout_rate'])
    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, f'model_cutoff{CUTOFF}.pth'),
        map_location='cpu'))
    model.eval()
    return model, scaler_X, scaler_y, label_encoders, categorical_info, num_cols

# ─────────────────────────────────────────────
# 构建固定类别的数值预测函数
# ─────────────────────────────────────────────
def build_numerical_predictor(model, scaler_X, scaler_y,
                               label_encoders, categorical_info,
                               fixed_cat_encoded):
    fixed_cat_t = torch.LongTensor(fixed_cat_encoded).unsqueeze(0)

    def predict_fn(x_num_scaled):
        n = x_num_scaled.shape[0]
        x_num_t = torch.FloatTensor(x_num_scaled)
        x_cat_t = fixed_cat_t.expand(n, -1)
        with torch.no_grad():
            pred_s = model(x_num_t, x_cat_t).numpy()
        pred = scaler_y.inverse_transform(pred_s).flatten()
        return pred

    return predict_fn

# ─────────────────────────────────────────────
# 全局绘图参数（NC风格）
# ─────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':        ['Arial', 'sans-serif'],
    'font.size':          11,
    'axes.labelsize':     11,
    'xtick.labelsize':    11,
    'ytick.labelsize':    11,
    'axes.linewidth':     0.8,
    'xtick.major.width':  0.8,
    'ytick.major.width':  0.8,
    'xtick.major.size':   3,
    'ytick.major.size':   3,
    'xtick.direction':    'in',
    'ytick.direction':    'in',
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# ─────────────────────────────────────────────
# SHAP可视化（NC风格）
# ─────────────────────────────────────────────
def plot_shap_summary(shap_values, feature_names, X_scaled_orig):
    if len(shap_values.shape) == 3:
        shap_values = shap_values.squeeze(axis=-1)

    # 缩写替换
    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names]

    # 按 mean|SHAP| 降序排列（最重要的在最上方）
    mean_abs_shap     = np.mean(np.abs(shap_values), axis=0)
    order             = np.argsort(-mean_abs_shap)
    shap_sorted       = shap_values[:, order]
    X_sorted          = X_scaled_orig[:, order]
    names_sorted      = [display_names[i] for i in order]
    importance_sorted = mean_abs_shap[order]

    # 散点配色：深蓝→青→浅黄→橙，与图2蓝橙体系一致
    scatter_cmap = mcolors.LinearSegmentedColormap.from_list(
        "nc_beeswarm", ['#4B6FA8', '#7EB6D4', '#F0E68C', '#E8913A']
    )
    # 背景bar：浅蓝灰，不抢眼
    bar_color = '#B8D4E8'

    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # ── 先画 beeswarm（shap接管ax，必须在twiny之前）─────────────
    plt.sca(ax)
    shap.summary_plot(
        shap_sorted,
        X_sorted,
        feature_names=names_sorted,
        show=False,
        plot_size=None,
        color_bar_label='Feature value',
        cmap=scatter_cmap,
        max_display=len(names_sorted),
        plot_type='dot',
        alpha=0.85,
    )
    ax = plt.gca()

    # ── 再创建顶部双轴，画 mean|SHAP| bar ───────────────────────
    ax2 = ax.twiny()
    n   = len(names_sorted)
    # shap.summary_plot: y=n-1 是最重要特征（最上），y=0 是最不重要（最下）
    # importance_sorted[0] 是最重要，对应 y=n-1
    for i in range(n):
        ax2.barh(
            n - 1 - i,
            importance_sorted[i],
            color=bar_color,
            alpha=0.40,
            height=0.75,
            zorder=1,
        )
    ax2.set_xlim(0, importance_sorted.max() * 1.25)
    ax2.set_ylim(ax.get_ylim())  # 与散点图y轴对齐
    ax2.set_xlabel('mean(|SHAP value|)', fontsize=11, color='#444444')
    ax2.tick_params(axis='x', labelsize=11, colors='#444444')
    ax2.spines['top'].set_linewidth(0.8)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)

    # 散点大小统一、zorder置顶
    for coll in ax.collections:
        coll.set_sizes([40])
        coll.set_zorder(3)

    # ── 坐标轴 ───────────────────────────────────────────────
    shap_min = shap_sorted.min()
    shap_max = shap_sorted.max()
    ax.set_xlim(shap_min - abs(shap_min) * 0.08,
                shap_max + abs(shap_max) * 0.08)

    ax.set_xlabel('SHAP value (impact on model output)', fontsize=11)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)

    # 四边封闭，细线
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.8)

    # 零线
    ax.axvline(0, color='#888888', linewidth=0.8, zorder=2)

    # 行间淡横线
    for i in range(n):
        ax.axhline(i, color='#E0E0E0', linewidth=0.4, zorder=0)

    # colorbar 字体
    if len(fig.axes) > 2:
        cbar_ax = fig.axes[-1]
        cbar_ax.tick_params(labelsize=11)
        cbar_ax.set_ylabel('Feature value', fontsize=11)

    plt.tight_layout()

    importance_df = pd.DataFrame({
        'Feature':        [feature_names[i] for i in order],
        'Display':        names_sorted,
        'Mean_Abs_SHAP':  importance_sorted,
    })
    return fig, importance_df

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    set_seed(42)
    print("Loading data...")
    df = load_data()
    tr_df = df[df['year'] < CUTOFF].copy()
    print(f"Train set: {len(tr_df)} samples")

    print("Loading model and preprocessors...")
    (model, scaler_X, scaler_y,
     label_encoders, categorical_info, num_cols) = load_model_and_preprocessors()

    # 过滤掉元素成分列，只保留物理描述符
    num_cols_phys = [c for c in num_cols if c not in ELEMENT_COLS]
    print(f"Physical descriptor cols ({len(num_cols_phys)}): {num_cols_phys}")

    # 数值特征缩放
    col_idx         = [num_cols.index(c) for c in num_cols_phys]
    X_num_tr        = tr_df[num_cols].values.astype(float)
    X_num_tr_s_full = scaler_X.transform(X_num_tr)
    X_num_tr_s      = X_num_tr_s_full[:, col_idx]

    # 类别特征：取训练集各列众数作为固定值
    fixed_cat = []
    for col in CATEGORICAL_COLUMNS:
        le       = label_encoders[col]
        mode_val = tr_df[col].astype(str).mode()[0]
        fixed_cat.append(int(le.transform([mode_val])[0]))
    fixed_cat_encoded = np.array(fixed_cat)
    print(f"Fixed categorical features (mode): "
          f"{dict(zip(CATEGORICAL_COLUMNS, fixed_cat_encoded))}")

    # 预测函数：物理描述符可变，元素列固定为训练集均值
    X_num_tr_full_mean = X_num_tr_s_full.mean(axis=0)

    def predict_fn_phys(x_phys_scaled):
        n      = x_phys_scaled.shape[0]
        x_full = np.tile(X_num_tr_full_mean, (n, 1))
        x_full[:, col_idx] = x_phys_scaled
        x_num_t = torch.FloatTensor(x_full)
        x_cat_t = torch.LongTensor(fixed_cat_encoded).unsqueeze(0).expand(n, -1)
        with torch.no_grad():
            pred_s = model(x_num_t, x_cat_t).numpy()
        return scaler_y.inverse_transform(pred_s).flatten()

    # SHAP
    n_bg       = min(100, len(X_num_tr_s))
    bg_idx     = np.random.choice(len(X_num_tr_s), n_bg, replace=False)
    background = X_num_tr_s[bg_idx]

    print(f"Computing SHAP values (background={n_bg}, explain={n_bg} samples)...")
    explainer   = shap.KernelExplainer(predict_fn_phys, background)
    shap_values = explainer.shap_values(X_num_tr_s[:100], nsamples=200)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    print("Generating SHAP plot...")
    fig, importance_df = plot_shap_summary(
        shap_values, num_cols_phys, X_num_tr_s[:100])

    out_svg = os.path.join(OUTPUT_DIR, 'shap_combinednet.svg')
    out_csv = os.path.join(OUTPUT_DIR, 'shap_importance.csv')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    importance_df.to_csv(out_csv, index=False)
    # 导出每个样本的 SHAP 值和特征值（用于重画和 Source Data）
    pd.DataFrame(shap_values, columns=num_cols_phys).to_csv(
        os.path.join(OUTPUT_DIR, 'shap_values_per_sample.csv'), index=False)
    pd.DataFrame(X_num_tr_s[:100], columns=num_cols_phys).to_csv(
        os.path.join(OUTPUT_DIR, 'shap_feature_values_scaled.csv'), index=False)

    print(f"\nSaved → {out_svg}")
    print(f"Saved → {out_csv}")
    print("\nFeature importance ranking:")
    for _, row in importance_df.iterrows():
        print(f"  {row['Display']:20s}  mean|SHAP| = {row['Mean_Abs_SHAP']:.4f}")
    print("\nDone.")