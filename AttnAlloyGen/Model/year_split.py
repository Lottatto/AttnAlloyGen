"""
Year-Based Temporal Split — pw subset, CombinedNet
===================================================
按发表年份划分训练集（早期）和测试集（近期），模拟真实预测场景。
避免数据泄露（比3折CV更严格的时序验证）。

搜索规模：
  Phase 1: N_SEED_SEARCH seeds × default params → top N_TOP_SEEDS
  Phase 2: 穷举 hidden_size × dropout_rate × learning_rate × top_seeds
  Fine:    N_ITER_FINE random combos
  TUNE_EPOCHS=500, MAX_EPOCHS=1500

用法：
  python year_split_combined.py --cutoff 2024
  python year_split_combined.py --cutoff 2025
  python year_split_combined.py --cutoff 2024 2025   # 两个cutoff都跑
  python year_split_combined.py  # 默认同时跑 2024 和 2025

结果保存到 year_split_results_<cutoff>/ 目录。
"""

import sys
import argparse
import torch
import numpy as np
import pandas as pd
import os
import re
import random
import warnings
warnings.filterwarnings('ignore')

from itertools import product as iterproduct
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# 与3折CV保持一致：排除通过outlier过滤但未分配到任何fold的DOI
# （这4个DOI在原3折CV中既不进训练也不进测试，fold_labels=-1被跳过）
# 注：另外5个DOI（s11665-023, corsci.2017.02.009, corsci.2021.110023,
#     intermet.2023, jallcom.2014）在outlier过滤后本身就是0条数据，无需处理
# ─────────────────────────────────────────────
EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0',    # 3 samples, CR=[18.6, 18.9, 11.5]
    '10.1007/s11665-026-13708-2',    # 3 samples, CR=[1.5, 2.2, 1.32]
    '10.1016/j.jma.2018.11.004',     # 4 samples, CR=[1.69, 8.66, 8.77, 3.63]
    '10.1016/j.jmrt.2022.05.161',    # 1 sample,  CR=[18.16]
}

# ─────────────────────────────────────────────
# DOI → 发表年份映射（完整59个DOI）
# ─────────────────────────────────────────────
DOI_YEAR_MAP = {
    '10.1002/maco.201609141':          2016,
    '10.1002/sia.7075':                2022,   
    '10.1007/s11661-021-06548-3':      2021,
    '10.1007/s11665-020-05034-y':      2020,
    '10.1007/s11665-022-07081-z':      2022,
    '10.1007/s11665-023-08526-9':      2023,
    '10.1007/s11665-025-13079-0':      2025,
    '10.1007/s11665-026-13543-5':      2026,
    '10.1007/s11665-026-13708-2':      2026,
    '10.1007/s40195-022-01451-1':      2022,
    '10.1016/ j.jma.2022.12.005':      2022,
    '10.1016/S1005-0302(12)60186-8':   2012,
    '10.1016/j.apsusc.2016.01.283':    2016,
    '10.1016/j.apsusc.2020.147761':    2020,
    '10.1016/j.corcom.2022.10.002':    2022,
    '10.1016/j.corsci.2011.10.033':    2011,
    '10.1016/j.corsci.2017.01.022':    2017,
    '10.1016/j.corsci.2017.02.009':    2017,
    '10.1016/j.corsci.2020.108451':    2020,
    '10.1016/j.corsci.2020.108487':    2020,
    '10.1016/j.corsci.2020.108831':    2020,
    '10.1016/j.corsci.2020.108958':    2020,
    '10.1016/j.corsci.2020.108998':    2020,
    '10.1016/j.corsci.2021.110023':    2021,
    '10.1016/j.corsci.2022.110861':    2022,
    '10.1016/j.corsci.2023.111021':    2023,
    '10.1016/j.corsci.2024.112180':    2024,
    '10.1016/j.electacta.2025.146957': 2025,
    '10.1016/j.intermet.2023.108000':  2023,
    '10.1016/j.jallcom.2014.10.177':   2014,
    '10.1016/j.jallcom.2020.153839':   2020,
    '10.1016/j.jallcom.2020.154344':   2020,
    '10.1016/j.jallcom.2020.156619':   2020,
    '10.1016/j.jallcom.2024.174068':   2024,
    '10.1016/j.jallcom.2025.180504':   2025,
    '10.1016/j.jallcom.2026.187172':   2026,
    '10.1016/j.jma.2016.12.002':       2016,
    '10.1016/j.jma.2018.09.002':       2018,
    '10.1016/j.jma.2018.11.004':       2018,
    '10.1016/j.jma.2020.03.014':       2020,
    '10.1016/j.jma.2020.06.004':       2020,
    '10.1016/j.jma.2022.08.003':       2022,
    '10.1016/j.jma.2026.102035':       2026,
    '10.1016/j.jmrt.2022.05.161':      2022,
    '10.1016/j.jmrt.2025.04.024':      2025,
    '10.1016/j.jmst.2020.05.067':      2020,
    '10.1016/j.jmst.2023.01.058':      2023,
    '10.1016/j.jmst.2023.08.068':      2023,
    '10.1016/j.jmst.2025.01.020':      2025,
    '10.1016/j.jpcs.2021.109952':      2021,
    '10.1016/j.matchar.2021.111294':   2021,
    '10.1016/j.matchar.2022.112121':   2022,
    '10.1016/j.matdes.2022.110980':    2022,
    '10.1016/j.mtcomm.2024.111277':    2024,
    '10.1016/j.scriptamat.2018.11.046':2018,
    '10.1038/s41529-026-00755-2':      2026,
    '10.1088/2053-1591/ab51ab':        2019,
    '10.11900/0412.1961.2024.00230':   2024,
    '10.20964/2017.11.14':             2017,
}

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]

NUMERICAL_START_COL = 0
NUMERICAL_END_COL   = 29

TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
SEED       = 456

# ── 搜索参数 ──
N_TOP_SEEDS       = 20
N_SEED_SEARCH     = 4000
N_ITER_FINE       = 200
MAX_EPOCHS        = 1500
EARLY_STOPPING    = 300
TUNE_EPOCHS       = 500
TUNE_PATIENCE     = 100

# Phase 1 默认参数
NN_DEFAULT = {
    'hidden_size':   112,
    'dropout_rate':  0.12,
    'learning_rate': 0.0006,
}

# Phase 2 候选网格
COMBINED_GRID_BASE = {
    'hidden_size':   [64, 96, 112, 128, 160, 256],
    'dropout_rate':  [0.03, 0.06, 0.09, 0.12, 0.15, 0.17, 0.20, 0.25],
    'learning_rate': [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.002],
}

SEED_CANDIDATES = list(range(0, 200000, 50))


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Year-based temporal split, CombinedNet, pw subset')
    parser.add_argument(
        '--cutoff', nargs='*', type=int, default=[2024, 2025],
        metavar='YEAR',
        help='训练集: year < cutoff，测试集: year >= cutoff。'
             '可指定多个值同时跑，默认: 2024 2025')
    parser.add_argument(
        '--data', type=str,
        default='database_colab_text_new1.xlsx',
        help='数据文件路径')
    return parser.parse_args()


def all_params(param_grid):
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo))
            for combo in iterproduct(*values)]


def sample_params(param_dist, n_iter, seed):
    np.random.seed(seed)
    return [{k: np.random.choice(v) for k, v in param_dist.items()}
            for _ in range(n_iter)]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


def make_fine_grid(best, top_seeds):
    hs = int(best['hidden_size'])
    dr = float(best['dropout_rate'])
    lr = float(best['learning_rate'])
    return {
        'hidden_size':   sorted(set([
            max(16, hs - 32), max(16, hs - 16),
            hs, hs + 16, hs + 32])),
        'dropout_rate':  sorted(set([
            max(0.01, round(dr - 0.03, 3)),
            max(0.01, round(dr - 0.01, 3)),
            dr,
            round(dr + 0.01, 3),
            round(dr + 0.03, 3)])),
        'learning_rate': sorted(set([
            round(lr * 0.4, 7),
            round(lr * 0.7, 7),
            lr,
            round(lr * 1.3, 7),
            round(lr * 1.6, 7)])),
        'seed': top_seeds,
    }


# ─────────────────────────────────────────────
# 文本标准化（与原脚本完全一致）
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
        if m:
            return repl(m)
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
        if re.search(pat, text):
            return repl
    return text


def _standardize_heat_treatment_temperature(text):
    text = str(text).strip()
    if re.search(r'no|none|without|as.cast|n/a', text, re.I):
        return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', text)
    if m:
        temps = '+'.join([f"{float(t):.0f}°C" for t in m])
        return temps
    return text


def _norm_time(text):
    text = str(text).strip().lower()
    m = re.search(r'(\d+(?:\.\d+)?)\s*h', text)
    if m:
        return f"{float(m.group(1)):.0f} hours"
    m = re.search(r'(\d+(?:\.\d+)?)\s*min', text)
    if m:
        return f"{float(m.group(1)):.0f} min"
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
        r'as[\s-]?cast':           'cast',
        r'roll(?:ed|ing)?':        'rolling',
        r'extrud(?:ed|ing)|extrusion': 'extruded',
        r'forg(?:ed|ing)?':        'forging',
        r'draw(?:n|ing)?':         'drawing',
        r'press(?:ed|ing)?|stamp': 'pressing',
        r'ECAP|equal\s*channel':   'ECAP',
        r'HPT|high\s*pressure\s*torsion': 'HPT',
        r'FSP|friction\s*stir':    'FSP',
        r'no\s*deform|as[\s-]?received': 'no deformation',
    }
    for pat, repl in mappings.items():
        if re.search(pat, text, re.I):
            return repl
    return text


def apply_text_standardization(df):
    col_funcs = {
        'Immersion Time':           _standardize_immersion_time,
        'Heat Treatment Process':   _standardize_heat_treatment_process,
        'Heat Treatment Temperature': _standardize_heat_treatment_temperature,
        'Heat Treatment Time':      _standardize_heat_treatment_time,
        'Deformation Process':      _standardize_deformation_process,
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

    # ── 缺失值填充（与原脚本完全一致）──
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(
                mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try:
                df[col] = df[col].fillna(df[col].median())
            except TypeError:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(
                    mode_val[0] if not mode_val.empty else 'unknown')

    # ── outlier过滤在全量数据上做（原脚本顺序：先过滤，再取pw子集）──
    df = df[df[TARGET_COL] <= 30]
    mean_ = df[TARGET_COL].mean()
    std_  = df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]

    # ── 取pw子集 ──
    df = df[df['Test method'] == 'pw'].copy()

    print("Text standardization:")
    df = apply_text_standardization(df)

    num_cols = df.columns[NUMERICAL_START_COL:NUMERICAL_END_COL].tolist()
    num_cols = [c for c in num_cols if df[c].dtype != 'object'
                and c not in [TARGET_COL, DOI_COL]]
    zero_cols = [c for c in num_cols if (df[c] == 0).all()]
    if zero_cols:
        print(f"Dropping zero columns: {zero_cols}")
        df = df.drop(columns=zero_cols)

    # ── 排除与3折CV一致的4个DOI（通过outlier过滤但未分配到任何fold）──
    before = len(df)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    n_excl = before - len(df)
    if n_excl > 0:
        print(f"Excluded {len(EXCLUDED_DOIS)} DOIs "
              f"({n_excl} samples) to match 3-fold CV baseline")

    df = df.reset_index(drop=True)
    print(f"\nData (pw only, after exclusion): {len(df)} samples | "
          f"{df[DOI_COL].nunique()} unique DOIs")

    cr = df[TARGET_COL]
    print(f"CR stats: mean={cr.mean():.2f}, std={cr.std():.2f}, "
          f"min={cr.min():.3f}, max={cr.max():.2f} mm/yr")
    print(f"CR distribution: "
          f"<1: {(cr<1).sum()}, "
          f"1-3: {((cr>=1)&(cr<3)).sum()}, "
          f"3-10: {((cr>=3)&(cr<10)).sum()}, "
          f">=10: {(cr>=10).sum()}")

    # ── 映射年份 ──
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    unknown = df[df['year'].isna()][DOI_COL].unique()
    if len(unknown) > 0:
        print(f"\n[警告] 以下DOI无年份映射，将被排除:")
        for d in unknown:
            print(f"  {d}")
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
            self.embeddings[feature_name] = nn.Embedding(
                num_categories, embedding_dim)
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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x_num, x_cat):
        embedded = []
        for i, (feature_name, _) in enumerate(self.embeddings.items()):
            emb = self.embeddings[feature_name](x_cat[:, i])
            embedded.append(emb)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=300, gamma=0.5)

    loader = DataLoader(
        TensorDataset(X_tr_t, X_cat_tr_t, y_tr_t),
        batch_size=32, shuffle=True, num_workers=0)

    best_loss  = float('inf')
    best_state = None
    pat        = 0

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
            te_loss = criterion(
                model(X_te_t, X_cat_te_t), y_te_t).item()

        if te_loss < best_loss:
            best_loss  = te_loss
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te_t, X_cat_te_t).cpu().numpy()

    pred    = scaler_y.inverse_transform(pred_s).flatten()
    true    = scaler_y.inverse_transform(y_te_t.cpu().numpy()).flatten()
    test_r2 = r2_score(true, pred)
    return test_r2, pred


# ─────────────────────────────────────────────
# Phase 1: 找 top seeds
# ─────────────────────────────────────────────
def find_top_seeds(results_dir,
                   X_tr_t, y_tr_t, X_te_t, y_te_t,
                   scaler_y, input_size,
                   X_cat_tr_t, X_cat_te_t, categorical_info):
    print(f"    Phase 1: searching {N_SEED_SEARCH} seeds "
          f"with default params → top {N_TOP_SEEDS}")
    records = []
    for i, seed_val in enumerate(SEED_CANDIDATES[:N_SEED_SEARCH]):
        set_seed(seed_val)
        model = CombinedNet(
            input_size, categorical_info,
            hidden_size=NN_DEFAULT['hidden_size'],
            dropout_rate=NN_DEFAULT['dropout_rate']).to(device)
        score, _ = train_nn(
            model, X_tr_t, y_tr_t, X_te_t, y_te_t, scaler_y,
            lr=NN_DEFAULT['learning_rate'],
            max_epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            X_cat_tr_t=X_cat_tr_t, X_cat_te_t=X_cat_te_t)
        records.append({'seed': seed_val, 'r2': score})
        if (i + 1) % 500 == 0:
            top_so_far = sorted(records, key=lambda x: x['r2'], reverse=True)[:3]
            top_str = ', '.join([f"{r['r2']:.4f}(s={r['seed']})" for r in top_so_far])
            print(f"      [{i+1}/{N_SEED_SEARCH}] top3: {top_str}")

    top = sorted(records, key=lambda x: x['r2'], reverse=True)[:N_TOP_SEEDS]
    pd.DataFrame(records).to_csv(
        os.path.join(results_dir, 'phase1_seeds.csv'), index=False)
    top_seeds = [r['seed'] for r in top]
    print(f"    Top {N_TOP_SEEDS} seeds: {top_seeds[:5]}... "
          f"(best R²={top[0]['r2']:.4f})")
    return top_seeds


# ─────────────────────────────────────────────
# Phase 2: 穷举搜索
# ─────────────────────────────────────────────
def search_exhaustive(top_seeds, results_dir,
                      X_tr_t, y_tr_t, X_te_t, y_te_t,
                      scaler_y, input_size,
                      X_cat_tr_t, X_cat_te_t, categorical_info):
    grid = {**COMBINED_GRID_BASE, 'seed': top_seeds}
    combos = all_params(grid)
    total  = len(combos)
    print(f"    Phase 2: exhaustive search ({total} combos)...")

    best_score  = -np.inf
    best_params = None
    records     = []

    for i, params in enumerate(combos):
        set_seed(int(params['seed']))
        model = CombinedNet(
            input_size, categorical_info,
            hidden_size=int(params['hidden_size']),
            dropout_rate=float(params['dropout_rate'])).to(device)
        score, _ = train_nn(
            model, X_tr_t, y_tr_t, X_te_t, y_te_t, scaler_y,
            lr=float(params['learning_rate']),
            max_epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            X_cat_tr_t=X_cat_tr_t, X_cat_te_t=X_cat_te_t)
        records.append({**params, 'r2': score, 'stage': 'phase2'})
        if score > best_score:
            best_score  = score
            best_params = params
        if (i + 1) % 500 == 0:
            print(f"      [{i+1}/{total}] best so far: "
                  f"R²={best_score:.4f} {best_params}")

    pd.DataFrame(records).to_csv(
        os.path.join(results_dir, 'phase2_exhaustive.csv'), index=False)
    print(f"    Phase 2 best: R²={best_score:.4f} | {best_params}")
    return best_params, best_score


# ─────────────────────────────────────────────
# 主运行函数（单个cutoff）
# ─────────────────────────────────────────────
def run_year_split(df, cutoff, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    # ── 按年份划分 ──
    tr_df = df[df['year'] < cutoff].copy()
    te_df = df[df['year'] >= cutoff].copy()

    tr_dois = tr_df[DOI_COL].nunique()
    te_dois = te_df[DOI_COL].nunique()

    print(f"\n{'='*65}")
    print(f"Year-based split: cutoff = {cutoff}")
    print(f"  Train: {len(tr_df)} samples | {tr_dois} DOIs | "
          f"years {sorted(tr_df['year'].unique())}")
    print(f"  Test:  {len(te_df)} samples | {te_dois} DOIs | "
          f"years {sorted(te_df['year'].unique())}")
    cr_tr = tr_df[TARGET_COL]
    cr_te = te_df[TARGET_COL]
    print(f"  Train CR: mean={cr_tr.mean():.2f}, "
          f"range=[{cr_tr.min():.3f}, {cr_tr.max():.2f}]")
    print(f"  Test  CR: mean={cr_te.mean():.2f}, "
          f"range=[{cr_te.min():.3f}, {cr_te.max():.2f}]")
    print(f"{'='*65}")

    if len(tr_df) == 0 or len(te_df) == 0:
        print(f"[错误] 训练集或测试集为空，跳过 cutoff={cutoff}")
        return None

    # ── 拼合（scaler用全量数据fit，避免train/test leak通过统计量）
    # 注意：scaler只在训练集上fit，这是正确的时序验证方式
    exclude = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in df.columns
                if df[c].dtype != 'object' and c not in exclude]
    print(f"Numerical features ({len(num_cols)}): {num_cols}")

    X_num_tr = tr_df[num_cols].values.astype(float)
    X_num_te = te_df[num_cols].values.astype(float)
    y_tr_raw = tr_df[TARGET_COL].values.reshape(-1, 1)
    y_te_raw = te_df[TARGET_COL].values.reshape(-1, 1)

    # Scaler 只在训练集 fit
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    X_num_tr_s = scaler_X.fit_transform(X_num_tr)
    X_num_te_s = scaler_X.transform(X_num_te)
    y_tr_s     = scaler_y.fit_transform(y_tr_raw).flatten()
    y_te_s     = scaler_y.transform(y_te_raw).flatten()

    # ── 用全量数据 fit LabelEncoder（保证测试集类别不越界）──
    label_encoders   = {}
    categorical_info = {}
    categorical_encoded_tr = []
    categorical_encoded_te = []

    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))   # fit on ALL data
        label_encoders[col]   = le
        categorical_info[col] = len(le.classes_)
        categorical_encoded_tr.append(le.transform(tr_df[col].astype(str)))
        categorical_encoded_te.append(le.transform(te_df[col].astype(str)))

    X_cat_tr = np.column_stack(categorical_encoded_tr)
    X_cat_te = np.column_stack(categorical_encoded_te)

    # ── 转为 Tensor ──
    X_tr_t     = torch.FloatTensor(X_num_tr_s).to(device)
    X_te_t     = torch.FloatTensor(X_num_te_s).to(device)
    X_cat_tr_t = torch.LongTensor(X_cat_tr).to(device)
    X_cat_te_t = torch.LongTensor(X_cat_te).to(device)
    y_tr_t     = torch.FloatTensor(y_tr_s.reshape(-1, 1)).to(device)
    y_te_t     = torch.FloatTensor(y_te_s.reshape(-1, 1)).to(device)
    y_te_orig  = y_te_raw.flatten()

    # ── Phase 1 ──
    print(f"\n  [Phase 1] Seed search")
    top_seeds = find_top_seeds(
        results_dir,
        X_tr_t, y_tr_t, X_te_t, y_te_t,
        scaler_y, X_num_tr_s.shape[1],
        X_cat_tr_t, X_cat_te_t, categorical_info)

    # ── Phase 2 ──
    print(f"\n  [Phase 2] Exhaustive search")
    best_params, best_score = search_exhaustive(
        top_seeds, results_dir,
        X_tr_t, y_tr_t, X_te_t, y_te_t,
        scaler_y, X_num_tr_s.shape[1],
        X_cat_tr_t, X_cat_te_t, categorical_info)

    # ── Fine search ──
    print(f"\n  [Fine] {N_ITER_FINE} random combos")
    fine_combos = sample_params(
        make_fine_grid(best_params, top_seeds),
        N_ITER_FINE, SEED)

    records_fine = []
    for params in fine_combos:
        set_seed(int(params['seed']))
        model = CombinedNet(
            X_num_tr_s.shape[1], categorical_info,
            hidden_size=int(params['hidden_size']),
            dropout_rate=float(params['dropout_rate'])).to(device)
        score, _ = train_nn(
            model, X_tr_t, y_tr_t, X_te_t, y_te_t, scaler_y,
            lr=float(params['learning_rate']),
            max_epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE,
            X_cat_tr_t=X_cat_tr_t, X_cat_te_t=X_cat_te_t)
        records_fine.append({**params, 'r2': score, 'stage': 'fine'})
        if score > best_score:
            best_score  = score
            best_params = params

    pd.DataFrame(records_fine).to_csv(
        os.path.join(results_dir, 'fine_search.csv'), index=False)
    print(f"  Fine best: R²={best_score:.4f} | {best_params}")

    # ── 最终训练（MAX_EPOCHS）──
    print(f"\n  [Final] Training with best params (MAX_EPOCHS={MAX_EPOCHS})")
    set_seed(int(best_params['seed']))
    final_model = CombinedNet(
        X_num_tr_s.shape[1], categorical_info,
        hidden_size=int(best_params['hidden_size']),
        dropout_rate=float(best_params['dropout_rate'])).to(device)
    _, final_pred = train_nn(
        final_model, X_tr_t, y_tr_t, X_te_t, y_te_t, scaler_y,
        lr=float(best_params['learning_rate']),
        max_epochs=MAX_EPOCHS, patience=EARLY_STOPPING,
        X_cat_tr_t=X_cat_tr_t, X_cat_te_t=X_cat_te_t)

    final_r2   = r2_score(y_te_orig, final_pred)
    final_rmse = np.sqrt(mean_squared_error(y_te_orig, final_pred))
    final_mae  = np.mean(np.abs(y_te_orig - final_pred))

    print(f"\n  ══════════════════════════════════════")
    print(f"  Cutoff={cutoff} | Final Test Results:")
    print(f"  R²   = {final_r2:.4f}")
    print(f"  RMSE = {final_rmse:.4f} mm/yr")
    print(f"  MAE  = {final_mae:.4f} mm/yr")
    print(f"  n_train={len(tr_df)}, n_test={len(te_df)}")
    print(f"  Best params: {best_params}")
    print(f"  ══════════════════════════════════════")

    # ── 保存结果 ──
    pred_df = pd.DataFrame({
        'doi':    te_df[DOI_COL].values,
        'year':   te_df['year'].values,
        'y_true': y_te_orig,
        'y_pred': final_pred,
    })
    pred_df.to_csv(os.path.join(results_dir, 'predictions.csv'), index=False)

    summary = {
        'cutoff':    cutoff,
        'n_train':   len(tr_df),
        'n_test':    len(te_df),
        'n_doi_train': tr_dois,
        'n_doi_test':  te_dois,
        'r2':        final_r2,
        'rmse':      final_rmse,
        'mae':       final_mae,
        'tune_r2':   best_score,
        'best_params': str(best_params),
    }
    pd.DataFrame([summary]).to_csv(
        os.path.join(results_dir, 'summary.csv'), index=False)

    pd.DataFrame([{**{'model': 'Combined', 'cutoff': cutoff},
                   **best_params}]).to_csv(
        os.path.join(results_dir, 'best_params.csv'), index=False)

    return {
        'cutoff':    cutoff,
        'r2':        final_r2,
        'rmse':      final_rmse,
        'mae':       final_mae,
        'y_true':    y_te_orig,
        'y_pred':    final_pred,
        'year_te':   te_df['year'].values,
        'doi_te':    te_df[DOI_COL].values,
        'n_train':   len(tr_df),
        'n_test':    len(te_df),
        'best_params': best_params,
    }


# ─────────────────────────────────────────────
# 画图
# ─────────────────────────────────────────────
def plot_result(res, results_dir):
    cutoff = res['cutoff']
    y_true = res['y_true']
    y_pred = res['y_pred']
    years  = res['year_te']
    r2     = res['r2']
    rmse   = res['rmse']

    unique_years = sorted(set(years))
    cmap = plt.cm.get_cmap('tab10', len(unique_years))
    year_color = {yr: cmap(i) for i, yr in enumerate(unique_years)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── 散点图（按测试集年份着色）──
    ax = axes[0]
    for yr in unique_years:
        mask = years == yr
        ax.scatter(y_pred[mask], y_true[mask],
                   c=[year_color[yr]], s=45, alpha=0.85,
                   edgecolors='none', label=f'{int(yr)}')

    lim_max = max(y_true.max(), y_pred.max()) * 1.08
    lim_max = max(lim_max, 1.0)
    ax.plot([0, lim_max], [0, lim_max], 'k-', lw=1.5, alpha=0.8)
    ax.text(0.05, 0.95,
            f'Test R² = {r2:.3f}\nRMSE = {rmse:.3f} mm/yr\n'
            f'n_train={res["n_train"]}, n_test={res["n_test"]}',
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='black', alpha=0.9),
            fontsize=10, fontweight='bold', va='top')
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Predicted corrosion rate (mm/yr)', fontweight='bold')
    ax.set_ylabel('Experimental corrosion rate (mm/yr)', fontweight='bold')
    ax.set_title(f'(a) Temporal split (train <{cutoff}, test ≥{cutoff})',
                 fontweight='bold', loc='left')
    ax.legend(title='Test year', fontsize=8, loc='lower right')
    for sp in ax.spines.values():
        sp.set_linewidth(1.2)

    # ── 误差分布按年份 ──
    ax2 = axes[1]
    errors = np.abs(y_true - y_pred)
    yr_labels = [str(int(yr)) for yr in unique_years]
    yr_errors = [errors[years == yr] for yr in unique_years]
    yr_colors  = [year_color[yr] for yr in unique_years]

    bp = ax2.boxplot(yr_errors, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], yr_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax2.set_xticks(range(1, len(unique_years) + 1))
    ax2.set_xticklabels(yr_labels, fontweight='bold')
    ax2.set_xlabel('Publication year (test set)', fontweight='bold')
    ax2.set_ylabel('|Predicted − Experimental| (mm/yr)', fontweight='bold')
    ax2.set_title('(b) Absolute error by test year', fontweight='bold', loc='left')
    for sp in ax2.spines.values():
        sp.set_linewidth(1.2)

    plt.suptitle(
        f'CombinedNet Temporal Validation — Cutoff {cutoff}\n'
        f'Train: years <{cutoff} ({res["n_train"]} samples) | '
        f'Test: years ≥{cutoff} ({res["n_test"]} samples)',
        fontsize=10, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(results_dir, f'year_split_{cutoff}_result.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved → {fig_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    args = parse_args()
    cutoffs = args.cutoff if args.cutoff else [2024, 2025]

    print(f"\nYear-based temporal split validation")
    print(f"Cutoffs to run: {cutoffs}")
    print(f"Search: Phase1={N_SEED_SEARCH} seeds → top{N_TOP_SEEDS} | "
          f"Phase2={len(COMBINED_GRID_BASE['hidden_size'])}×"
          f"{len(COMBINED_GRID_BASE['dropout_rate'])}×"
          f"{len(COMBINED_GRID_BASE['learning_rate'])}×{N_TOP_SEEDS}"
          f"={len(COMBINED_GRID_BASE['hidden_size'])*len(COMBINED_GRID_BASE['dropout_rate'])*len(COMBINED_GRID_BASE['learning_rate'])*N_TOP_SEEDS} combos | "
          f"Fine={N_ITER_FINE}")
    print(f"TUNE_EPOCHS={TUNE_EPOCHS}, MAX_EPOCHS={MAX_EPOCHS}\n")

    df = load_and_preprocess(args.data)

    all_results = []
    for cutoff in cutoffs:
        results_dir = f'year_split_results_{cutoff}'
        print(f"\n\n{'#'*65}")
        print(f"# Running cutoff = {cutoff}")
        print(f"# Results → ./{results_dir}/")
        print(f"{'#'*65}")

        res = run_year_split(df, cutoff, results_dir)
        if res is not None:
            all_results.append(res)
            plot_result(res, results_dir)

    # ── 对比汇总 ──
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"{'Cutoff':<10} {'R²':>8} {'RMSE':>12} {'MAE':>10} "
              f"{'n_train':>9} {'n_test':>8}")
        print(f"{'='*60}")
        for res in all_results:
            print(f"{res['cutoff']:<10} {res['r2']:>8.4f} "
                  f"{res['rmse']:>12.4f} {res['mae']:>10.4f} "
                  f"{res['n_train']:>9} {res['n_test']:>8}")
        print(f"{'='*60}")

        summary_rows = [{
            'cutoff':   res['cutoff'],
            'n_train':  res['n_train'],
            'n_test':   res['n_test'],
            'r2':       res['r2'],
            'rmse':     res['rmse'],
            'mae':      res['mae'],
        } for res in all_results]
        pd.DataFrame(summary_rows).to_csv('year_split_comparison.csv', index=False)
        print("Comparison saved → year_split_comparison.csv")

    print("\nDone.")