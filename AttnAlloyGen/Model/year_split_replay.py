"""
Year-Split Replay — 直接加载最优参数，跳过搜索，复现最终结果
================================================================
Cutoff=2024: hidden=112, dr=0.06,  lr=0.0008, seed=143750
Cutoff=2025: hidden=112, dr=0.12,  lr=0.0006, seed=119700
"""

import torch
import numpy as np
import pandas as pd
import os
import re
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

matplotlib.rcParams.update({
    'font.family':       ['Arial', 'sans-serif'],
    'font.size':         8,
    'axes.linewidth':    0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'pdf.fonttype':      42,
    'ps.fonttype':       42,
    'svg.fonttype':      'none',
})

# 年份配色，与3-fold保持同一色系（NPG扩展）
YEAR_COLORS = ['#E64B35','#4DBBD5','#00A087','#3C5488',
               '#F39B7F','#8491B4','#91D1C2','#DC0000']

# ─────────────────────────────────────────────
# 最优参数
# ─────────────────────────────────────────────
BEST_PARAMS = {
    2024: {'hidden_size': 112, 'dropout_rate': 0.06,  'learning_rate': 0.0008, 'seed': 143750},
    2025: {'hidden_size': 112, 'dropout_rate': 0.12,  'learning_rate': 0.0006, 'seed': 119700},
}

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
NUMERICAL_START_COL = 0
NUMERICAL_END_COL   = 29
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
MAX_EPOCHS    = 1500
EARLY_STOPPING = 300
DATA_PATH = 'database_colab_text_new1.xlsx'

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(456)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────────
# 文本标准化
# ─────────────────────────────────────────────
def _standardize_immersion_time(text):
    text = str(text).strip().lower()
    patterns = [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)',  lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',   lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)',lambda m: f"{float(m.group(1)):.0f} min"),
        (r'(\d+(?:\.\d+)?)\s*s(?:econds?)?(?:\s|$|,)',lambda m: f"{float(m.group(1)):.0f} s"),
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
        r'cast(?:ed|ing)?|casted': 'cast',
        r'as[\s-]?cast': 'cast',
        r'roll(?:ed|ing)?': 'rolling',
        r'extrud(?:ed|ing)|extrusion': 'extruded',
        r'forg(?:ed|ing)?': 'forging',
        r'draw(?:n|ing)?': 'drawing',
        r'press(?:ed|ing)?|stamp': 'pressing',
        r'ECAP|equal\s*channel': 'ECAP',
        r'HPT|high\s*pressure\s*torsion': 'HPT',
        r'FSP|friction\s*stir': 'FSP',
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
            before = df[col].nunique()
            df[col] = df[col].apply(func)
            after  = df[col].nunique()
            print(f"  {col}: {before} → {after} unique values")
    return df

# ─────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────
def load_and_preprocess(path):
    df = pd.read_excel(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try: df[col] = df[col].fillna(df[col].median())
            except TypeError:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')

    df = df[df[TARGET_COL] <= 30]
    mean_ = df[TARGET_COL].mean()
    std_  = df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]
    df = df[df['Test method'] == 'pw'].copy()
    print("Text standardization:")
    df = apply_text_standardization(df)

    num_cols = df.columns[NUMERICAL_START_COL:NUMERICAL_END_COL].tolist()
    num_cols = [c for c in num_cols if df[c].dtype != 'object'
                and c not in [TARGET_COL, DOI_COL]]
    zero_cols = [c for c in num_cols if (df[c] == 0).all()]
    if zero_cols: df = df.drop(columns=zero_cols)

    before = len(df)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    print(f"Excluded 4 DOIs ({before - len(df)} samples)")
    df = df.reset_index(drop=True)
    print(f"Data (pw, after exclusion): {len(df)} samples | {df[DOI_COL].nunique()} DOIs")
    cr = df[TARGET_COL]
    print(f"CR: mean={cr.mean():.2f}, std={cr.std():.2f}, "
          f"min={cr.min():.3f}, max={cr.max():.2f} mm/yr")
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

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
# 训练函数
# ─────────────────────────────────────────────
def train_nn(model, X_tr_t, y_tr_t, X_te_t, y_te_t,
             scaler_y, lr, max_epochs, patience,
             X_cat_tr_t, X_cat_te_t):
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loader = DataLoader(TensorDataset(X_tr_t, X_cat_tr_t, y_tr_t),
                        batch_size=32, shuffle=True, num_workers=0)
    best_loss, best_state, pat = float('inf'), None, 0
    for epoch in range(max_epochs):
        model.train()
        for Xn, Xc, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xn, Xc), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            te_loss = criterion(model(X_te_t, X_cat_te_t), y_te_t).item()
        if te_loss < best_loss:
            best_loss  = te_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience: break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te_t, X_cat_te_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s).flatten()
    true = scaler_y.inverse_transform(y_te_t.cpu().numpy()).flatten()
    return r2_score(true, pred), pred

# ─────────────────────────────────────────────
# 推理
# ─────────────────────────────────────────────
def run_replay(df, cutoff, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    params = BEST_PARAMS[cutoff]
    tr_df = df[df['year'] < cutoff].copy()
    te_df = df[df['year'] >= cutoff].copy()
    print(f"\n{'='*60}")
    print(f"Cutoff={cutoff} | train={len(tr_df)} ({tr_df[DOI_COL].nunique()} DOIs) | "
          f"test={len(te_df)} ({te_df[DOI_COL].nunique()} DOIs)")
    print(f"Best params: {params}")

    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in df.columns
                if df[c].dtype != 'object' and c not in exclude]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_num_tr_s = scaler_X.fit_transform(tr_df[num_cols].values.astype(float))
    X_num_te_s = scaler_X.transform(te_df[num_cols].values.astype(float))
    y_tr_s     = scaler_y.fit_transform(tr_df[TARGET_COL].values.reshape(-1,1)).flatten()
    y_te_s_arr = scaler_y.transform(te_df[TARGET_COL].values.reshape(-1,1)).flatten()
    y_te_orig  = te_df[TARGET_COL].values

    categorical_info = {}
    label_encoders   = {}          # 新增：保存encoder对象供后续推理用
    cat_enc_tr, cat_enc_te = [], []
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        categorical_info[col] = len(le.classes_)
        label_encoders[col]   = le  # 新增
        cat_enc_tr.append(le.transform(tr_df[col].astype(str)))
        cat_enc_te.append(le.transform(te_df[col].astype(str)))

    X_tr_t     = torch.FloatTensor(X_num_tr_s).to(device)
    X_te_t     = torch.FloatTensor(X_num_te_s).to(device)
    X_cat_tr_t = torch.LongTensor(np.column_stack(cat_enc_tr)).to(device)
    X_cat_te_t = torch.LongTensor(np.column_stack(cat_enc_te)).to(device)
    y_tr_t     = torch.FloatTensor(y_tr_s.reshape(-1,1)).to(device)
    y_te_t     = torch.FloatTensor(y_te_s_arr.reshape(-1,1)).to(device)

    set_seed(int(params['seed']))
    model = CombinedNet(X_num_tr_s.shape[1], categorical_info,
                        hidden_size=int(params['hidden_size']),
                        dropout_rate=float(params['dropout_rate'])).to(device)
    r2, pred = train_nn(model, X_tr_t, y_tr_t, X_te_t, y_te_t, scaler_y,
                        lr=float(params['learning_rate']),
                        max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING,
                        X_cat_tr_t=X_cat_tr_t, X_cat_te_t=X_cat_te_t)

    rmse = np.sqrt(mean_squared_error(y_te_orig, pred))
    mae  = np.mean(np.abs(y_te_orig - pred))
    print(f"R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}")

    pd.DataFrame({
        'doi': te_df[DOI_COL].values,
        'year': te_df['year'].values,
        'y_true': y_te_orig,
        'y_pred': pred,
    }).to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

    # ── 保存模型权重和预处理对象（供 Part 1 生成质量评估使用）──
    import pickle
    torch.save(model.state_dict(),
               os.path.join(results_dir, f'model_cutoff{cutoff}.pth'))
    with open(os.path.join(results_dir, f'scaler_X_cutoff{cutoff}.pkl'), 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(os.path.join(results_dir, f'scaler_y_cutoff{cutoff}.pkl'), 'wb') as f:
        pickle.dump(scaler_y, f)
    with open(os.path.join(results_dir, f'label_encoders_cutoff{cutoff}.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(os.path.join(results_dir, f'categorical_info_cutoff{cutoff}.pkl'), 'wb') as f:
        pickle.dump(categorical_info, f)
    with open(os.path.join(results_dir, f'num_cols_cutoff{cutoff}.pkl'), 'wb') as f:
        pickle.dump(num_cols, f)
    print(f"Saved → {results_dir}/model_cutoff{cutoff}.pth  (+ scalers & encoders)")

    # ── 训练集预测（eval模式，用于散点图对比）──
    model.eval()
    with torch.no_grad():
        pred_tr_s = model(X_tr_t, X_cat_tr_t).cpu().numpy()
    pred_tr = scaler_y.inverse_transform(pred_tr_s).flatten()
    y_tr_orig = tr_df[TARGET_COL].values
    r2_tr = r2_score(y_tr_orig, pred_tr)
    print(f"Train R²={r2_tr:.4f}")

    return {'cutoff': cutoff, 'r2': r2, 'rmse': rmse, 'mae': mae,
            'y_true': y_te_orig, 'y_pred': pred,
            'year_te': te_df['year'].values,
            'n_train': len(tr_df), 'n_test': len(te_df),
            'y_tr_true': y_tr_orig, 'y_tr_pred': pred_tr, 'r2_tr': r2_tr}

# ─────────────────────────────────────────────
# 可视化 — NC风格双面板
# (a) 散点图（按年份着色）  (b) 按年份箱线图
# ─────────────────────────────────────────────
def plot_result(res, results_dir):
    cutoff = res['cutoff']
    y_true = res['y_true']
    y_pred = res['y_pred']
    years  = res['year_te']
    r2     = res['r2']
    rmse   = res['rmse']

    unique_years = sorted(set(years))
    year_color   = {yr: YEAR_COLORS[i % len(YEAR_COLORS)]
                    for i, yr in enumerate(unique_years)}

    all_vals = np.concatenate([y_true, y_pred])
    if res.get('y_tr_true') is not None:
        all_vals = np.concatenate([all_vals, res['y_tr_true'], res['y_tr_pred']])
    vmax = float(np.ceil(all_vals.max() * 1.08))
    vmax = max(vmax, 5.0)

    fig_w = 7.5 / 2.54
    fig_h = 7.5 / 2.54
    fig, ax_s = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # 训练集点
    y_tr_true = res.get('y_tr_true')
    y_tr_pred = res.get('y_tr_pred')
    r2_tr     = res.get('r2_tr')
    if y_tr_true is not None:
        ax_s.scatter(
            y_tr_pred, y_tr_true,
            color='#CCCCCC', s=20, alpha=0.45,
            edgecolors='none', zorder=2,
            label='Train',
        )

    # 测试集点
    for yr in unique_years:
        mask = years == yr
        ax_s.scatter(
            y_pred[mask], y_true[mask],
            color=year_color[yr],
            s=35, alpha=0.90,
            edgecolors='none',
            zorder=3,
            label=str(int(yr)),
        )

    ax_s.plot([0, vmax], [0, vmax],
              color='#555', linewidth=1.2,
              linestyle=(0, (4, 3)), zorder=2)

    ax_s.set_xlim(0, vmax)
    ax_s.set_ylim(0, vmax)
    ax_s.set_aspect('equal', adjustable='box')
    ax_s.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax_s.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
    ax_s.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax_s.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax_s.tick_params(labelsize=8, width=1.0, length=3.5)

    # 四边闭口
    for sp in ax_s.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(1.0)

    ax_s.grid(False)
    ax_s.set_xlabel('Predicted corrosion rate (mm\u2009yr$^{-1}$)',
                    fontsize=9, labelpad=4)
    ax_s.set_ylabel('Experimental corrosion rate (mm\u2009yr$^{-1}$)',
                    fontsize=9, labelpad=4)

    # R²和RMSE写在右侧轴外
    stat_lines = []
    if r2_tr is not None:
        stat_lines.append(f'Train $R^2$ = {r2_tr:.3f}')
    stat_lines.append(f'Test $R^2$ = {r2:.3f}')
    stat_lines.append(f'RMSE = {rmse:.3f} mm\u2009yr$^{{-1}}$')
    ax_s.text(1.04, 0.5, '\n'.join(stat_lines),
              transform=ax_s.transAxes,
              fontsize=8, va='center', ha='left',
              linespacing=1.8,
              rotation=270 if False else 0)

    # 图例
    ax_s.legend(title='Test year', fontsize=7.5,
                title_fontsize=8,
                loc='upper left', frameon=False,
                labelspacing=0.35, handletextpad=0.4,
                markerscale=1.1)

    # ── 保存 ──────────────────────────────────────────────────────────────
    os.makedirs(results_dir, exist_ok=True)
    out_png = os.path.join(results_dir, f'year_split_{cutoff}_nc.png')
    out_svg = os.path.join(results_dir, f'year_split_{cutoff}_nc.svg')
    fig.savefig(out_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(out_svg, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Saved → {out_png}")
    print(f"Saved → {out_svg}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', nargs='*', type=int, default=[2024, 2025])
    parser.add_argument('--data', type=str, default=DATA_PATH)
    args = parser.parse_args()

    print(f"\nYear-Split Replay (hardcoded best params)")
    df = load_and_preprocess(args.data)

    all_results = []
    for cutoff in args.cutoff:
        if cutoff not in BEST_PARAMS:
            print(f"[错误] cutoff={cutoff} 无对应最优参数，跳过")
            continue
        res = run_replay(df, cutoff, f'year_split_results_{cutoff}')
        plot_result(res, f'year_split_results_{cutoff}')
        all_results.append(res)

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Cutoff':<10} {'R²':>8} {'RMSE':>12} {'MAE':>10} "
              f"{'n_train':>9} {'n_test':>8}")
        print(f"{'='*60}")
        for res in all_results:
            print(f"{res['cutoff']:<10} {res['r2']:>8.4f} "
                  f"{res['rmse']:>12.4f} {res['mae']:>10.4f} "
                  f"{res['n_train']:>9} {res['n_test']:>8}")

    print("\nDone.")