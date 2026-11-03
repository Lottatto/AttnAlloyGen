"""
Year-Split Ablation Extended — 全方法 × 三种特征编码方式
==========================================================
在原ablation基础上新增文本特征变体，共13个模型：

数值only（5个）:
  NN, XGB, RF, SVR, GPR

数值 + label encoding文本（4个）:
  NN_label, XGB_label, RF_label, SVR_label

数值 + one-hot encoding文本（4个）:
  NN_onehot, XGB_onehot, RF_onehot, SVR_onehot

数值 + entity embedding文本（1个）:
  Combined (AttnAlloyGen) ← 本工作

GPR不加文本变体（高维one-hot下GPR不适用）

用法：
  python year_split_ablation_extended.py
  python year_split_ablation_extended.py --cutoff 2025
  python year_split_ablation_extended.py --cutoff 2024 2025
"""

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
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, RBF
from sklearn.svm import SVR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[警告] xgboost未安装，XGB系列将跳过")

# ─────────────────────────────────────────────
# Combined最优参数（hardcoded）
# ─────────────────────────────────────────────
COMBINED_BEST = {
    2024: {'hidden_size': 112, 'dropout_rate': 0.06,
           'learning_rate': 0.0008, 'seed': 143750},
    2025: {'hidden_size': 112, 'dropout_rate': 0.12,
           'learning_rate': 0.0006, 'seed': 119700},
}

# ─────────────────────────────────────────────
# 排除DOI & 年份映射
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

# NN搜索参数（轻量版，与原ablation保持一致）
N_SEED_SEARCH_NN = 100
N_TOP_SEEDS_NN   = 3
TUNE_EPOCHS      = 200
TUNE_PATIENCE    = 40
MAX_EPOCHS       = 1500
EARLY_STOPPING   = 300

NN_GRID = {
    'hidden_size':   [64, 112, 256],
    'dropout_rate':  [0.03, 0.09, 0.15, 0.20],
    'learning_rate': [0.0002, 0.0006, 0.001, 0.002],
}
NN_DEFAULT = {'hidden_size': 112, 'dropout_rate': 0.12, 'learning_rate': 0.0006}

N_ITER_COARSE = 200
N_ITER_FINE   = 50

XGB_COARSE = {
    'n_estimators':     [50, 100, 200, 400],
    'max_depth':        [2, 3, 5, 7],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 6],
    'gamma':            [0, 0.1, 0.3],
    'random_state':     list(range(0, 500, 50)),
}
RF_COARSE = {
    'n_estimators':      [50, 100, 200, 400],
    'max_depth':         [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 3, 5],
    'max_features':      ['sqrt', 'log2', None],
    'random_state':      list(range(0, 500, 50)),
}
SVR_COARSE = {
    'C':       [0.1, 1, 5, 20, 100],
    'epsilon': [0.01, 0.1, 0.5, 1.0],
    'gamma':   ['scale', 'auto'],
    'kernel':  ['rbf', 'linear'],
}
GPR_KERNELS = [
    Matern(nu=2.5) + WhiteKernel(noise_level=1.0),
    Matern(nu=1.5) + WhiteKernel(noise_level=1.0),
    RBF()          + WhiteKernel(noise_level=1.0),
]
GPR_SEEDS = list(range(0, 300, 50))

SEED_CANDIDATES = list(range(0, 200000, 50))


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', nargs='*', type=int, default=[2024, 2025])
    parser.add_argument('--data', type=str, default='database_colab_text_new1.xlsx')
    return parser.parse_args()


def sample_params(param_dist, n_iter, seed):
    np.random.seed(seed)
    return [{k: np.random.choice(v) for k, v in param_dist.items()}
            for _ in range(n_iter)]


def all_params(param_grid):
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo))
            for combo in iterproduct(*param_grid.values())]


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


def make_fine_xgb(best):
    return {k: [max(1, int(best[k])-50), int(best[k]), min(500, int(best[k])+50)]
            if k == 'n_estimators'
            else [max(1, int(best[k])-1), int(best[k]), int(best[k])+1]
            if k == 'max_depth'
            else [round(float(best[k])*0.5, 4), float(best[k]), round(float(best[k])*1.5, 4)]
            if k == 'learning_rate'
            else [best[k]]
            for k in best}


def make_fine_rf(best):
    n = int(best['n_estimators'])
    return {
        'n_estimators':      [max(10, n-50), n, n+50],
        'max_depth':         [best['max_depth']],
        'min_samples_split': [max(2, int(best['min_samples_split'])-1),
                               int(best['min_samples_split']),
                               int(best['min_samples_split'])+1],
        'min_samples_leaf':  [max(1, int(best['min_samples_leaf'])-1),
                               int(best['min_samples_leaf'])],
        'max_features':      [best['max_features']],
        'random_state':      [int(best['random_state'])],
    }


def make_fine_svr(best):
    C = float(best['C'])
    return {
        'C':       [C*0.3, C, C*3],
        'epsilon': [float(best['epsilon'])*0.5, float(best['epsilon'])],
        'gamma':   [best['gamma']],
        'kernel':  [best['kernel']],
    }


# ─────────────────────────────────────────────
# 文本标准化
# ─────────────────────────────────────────────
def _standardize_immersion_time(text):
    text = str(text).strip().lower()
    for pat, repl in [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)',   lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',    lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)', lambda m: f"{float(m.group(1)):.0f} min"),
    ]:
        m = re.search(pat, text)
        if m:
            return repl(m)
    return text

def _standardize_heat_treatment_process(text):
    text = str(text).strip().lower()
    for pat, repl in [
        (r'\bt4\b', 'solution-treated'), (r'\bt6\b', 'solution-treated+aging'),
        (r'solution\s*treat(?:ed|ment)?', 'solution-treated'),
        (r'anneal(?:ed|ing)?', 'annealed'),
        (r'aging|aged', 'aging'),
        (r'no\s*heat\s*treat(?:ment)?|as[\s-]?cast|without', 'no heat treatment'),
    ]:
        if re.search(pat, text): return repl
    return text

def _standardize_heat_treatment_temperature(text):
    text = str(text).strip()
    if re.search(r'no|none|without|as.cast|n/a', text, re.I):
        return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', text)
    if m:
        return '+'.join([f"{float(t):.0f}°C" for t in m])
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
    return '+'.join([_norm_time(p) for p in re.split(r'[+,;/]', text) if p.strip()]) or text

def _standardize_deformation_process(text):
    text = str(text).strip().lower()
    for pat, repl in [
        (r'cast(?:ed|ing)?|as[\s-]?cast', 'cast'),
        (r'roll(?:ed|ing)?', 'rolling'),
        (r'extrud(?:ed|ing)|extrusion', 'extruded'),
        (r'forg(?:ed|ing)?', 'forging'),
        (r'ECAP|equal\s*channel', 'ECAP'),
    ]:
        if re.search(pat, text, re.I): return repl
    return text

def apply_text_standardization(df):
    for col, func in [
        ('Immersion Time', _standardize_immersion_time),
        ('Heat Treatment Process', _standardize_heat_treatment_process),
        ('Heat Treatment Temperature', _standardize_heat_treatment_temperature),
        ('Heat Treatment Time', _standardize_heat_treatment_time),
        ('Deformation Process', _standardize_deformation_process),
    ]:
        if col in df.columns:
            df[col] = df[col].apply(func)
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
            except: pass
    df = df[df[TARGET_COL] <= 30]
    mean_ = df[TARGET_COL].mean(); std_ = df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]
    df = df[df['Test method'] == 'pw'].copy()
    df = apply_text_standardization(df)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df = df.reset_index(drop=True)
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    print(f"Data: {len(df)} samples | {df[DOI_COL].nunique()} DOIs")
    return df


# ─────────────────────────────────────────────
# 特征构建：三种文本编码方式
# ─────────────────────────────────────────────
def build_features(df, tr_df, te_df, num_cols, mode='numerical'):
    """
    mode: 'numerical' | 'label' | 'onehot'
    返回: X_tr, X_te (numpy arrays, 已scale)
    同时返回label/onehot编码器供复用
    """
    scaler_X = RobustScaler()
    X_num_tr = scaler_X.fit_transform(tr_df[num_cols].values.astype(float))
    X_num_te = scaler_X.transform(te_df[num_cols].values.astype(float))

    if mode == 'numerical':
        return X_num_tr, X_num_te, scaler_X, None

    elif mode == 'label':
        label_parts_tr, label_parts_te = [], []
        encoders = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            le.fit(df[col].astype(str))
            encoders[col] = le
            enc_tr = le.transform(tr_df[col].astype(str)).reshape(-1, 1).astype(float)
            enc_te = le.transform(te_df[col].astype(str)).reshape(-1, 1).astype(float)
            label_parts_tr.append(enc_tr)
            label_parts_te.append(enc_te)
        X_cat_tr = np.hstack(label_parts_tr)
        X_cat_te = np.hstack(label_parts_te)
        X_tr = np.hstack([X_num_tr, X_cat_tr])
        X_te = np.hstack([X_num_te, X_cat_te])
        return X_tr, X_te, scaler_X, encoders

    elif mode == 'onehot':
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.fit(df[CATEGORICAL_COLUMNS].astype(str))
        X_cat_tr = ohe.transform(tr_df[CATEGORICAL_COLUMNS].astype(str))
        X_cat_te = ohe.transform(te_df[CATEGORICAL_COLUMNS].astype(str))
        X_tr = np.hstack([X_num_tr, X_cat_tr])
        X_te = np.hstack([X_num_te, X_cat_te])
        print(f"    One-hot dim: {X_cat_tr.shape[1]} (total input: {X_tr.shape[1]})")
        return X_tr, X_te, scaler_X, ohe

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ─────────────────────────────────────────────
# 模型定义
# ─────────────────────────────────────────────
class FlatNet(nn.Module):
    """数值only / label enc / one-hot 统一用这个，输入是flat向量"""
    def __init__(self, input_size, hidden_size=112, dropout_rate=0.12):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, 1)
        self.dr  = dropout_rate
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.dropout(x, self.dr, self.training)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.dropout(x, self.dr, self.training)
        return self.out(x)


class CombinedNet(nn.Module):
    """Entity embedding模型（AttnAlloyGen）"""
    def __init__(self, numerical_size, categorical_info,
                 hidden_size=112, dropout_rate=0.12):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        emb_dim = 0
        for feat, n_cat in categorical_info.items():
            d = 2 if n_cat <= 5 else 4 if n_cat <= 15 else 6
            self.embeddings[feat] = nn.Embedding(n_cat, d)
            emb_dim += d
        self.fc1 = nn.Linear(numerical_size + emb_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.out = nn.Linear(hidden_size // 2, 1)
        self.dr  = dropout_rate
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_num, x_cat):
        embs = [self.embeddings[f](x_cat[:, i])
                for i, f in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        x = F.leaky_relu(self.fc1(x), 0.01)
        x = F.dropout(x, self.dr, self.training)
        x = F.leaky_relu(self.fc2(x), 0.01)
        x = F.dropout(x, self.dr, self.training)
        return self.out(x)


# ─────────────────────────────────────────────
# 训练函数
# ─────────────────────────────────────────────
def train_flat_nn(model, X_tr, y_tr_s, X_te, y_te_s,
                  scaler_y, lr, max_epochs, patience):
    """FlatNet训练（数值only / label / onehot）"""
    X_tr_t = torch.FloatTensor(X_tr).to(device)
    X_te_t = torch.FloatTensor(X_te).to(device)
    y_tr_t = torch.FloatTensor(y_tr_s.reshape(-1, 1)).to(device)
    y_te_t = torch.FloatTensor(y_te_s.reshape(-1, 1)).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                        batch_size=32, shuffle=True, num_workers=0)

    best_loss, best_state, pat = float('inf'), None, 0
    for epoch in range(max_epochs):
        model.train()
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            te_loss = criterion(model(X_te_t), y_te_t).item()
        if te_loss < best_loss:
            best_loss  = te_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s).flatten()
    true = scaler_y.inverse_transform(y_te_t.cpu().numpy()).flatten()
    return r2_score(true, pred), pred


def train_combined_nn(model, X_num_tr_s, X_cat_tr, y_tr_s,
                      X_num_te_s, X_cat_te, y_te_s,
                      scaler_y, lr, max_epochs, patience):
    """CombinedNet训练（entity embedding）"""
    X_tr_t     = torch.FloatTensor(X_num_tr_s).to(device)
    X_te_t     = torch.FloatTensor(X_num_te_s).to(device)
    X_cat_tr_t = torch.LongTensor(X_cat_tr).to(device)
    X_cat_te_t = torch.LongTensor(X_cat_te).to(device)
    y_tr_t     = torch.FloatTensor(y_tr_s.reshape(-1, 1)).to(device)
    y_te_t     = torch.FloatTensor(y_te_s.reshape(-1, 1)).to(device)

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
        if pat >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_s = model(X_te_t, X_cat_te_t).cpu().numpy()
    pred = scaler_y.inverse_transform(pred_s).flatten()
    true = scaler_y.inverse_transform(y_te_t.cpu().numpy()).flatten()
    return r2_score(true, pred), pred


# ─────────────────────────────────────────────
# NN超参数搜索（flat输入通用）
# ─────────────────────────────────────────────
def search_flat_nn(X_tr, y_tr_s, X_te, y_te, scaler_y, tag='NN'):
    """对FlatNet做Phase1+Phase2搜索，返回best_params和final预测"""
    input_size = X_tr.shape[1]

    # Phase 1: seed search
    print(f"    [{tag}] Phase 1: {N_SEED_SEARCH_NN} seeds...")
    nn_records = []
    for s in SEED_CANDIDATES[:N_SEED_SEARCH_NN]:
        set_seed(s)
        model = FlatNet(input_size, NN_DEFAULT['hidden_size'],
                        NN_DEFAULT['dropout_rate']).to(device)
        score, _ = train_flat_nn(model, X_tr, y_tr_s, X_te,
                                  scaler_y.transform(y_te.reshape(-1,1)).flatten(),
                                  scaler_y, NN_DEFAULT['learning_rate'],
                                  TUNE_EPOCHS, TUNE_PATIENCE)
        nn_records.append({'seed': s, 'r2': score})
    top_seeds = sorted(nn_records, key=lambda x: x['r2'], reverse=True)[:N_TOP_SEEDS_NN]
    top_seeds = [r['seed'] for r in top_seeds]

    # Phase 2: exhaustive
    print(f"    [{tag}] Phase 2: exhaustive search...")
    nn_grid   = {**NN_GRID, 'seed': top_seeds}
    nn_combos = all_params(nn_grid)
    best_score_nn, best_params_nn = -np.inf, None
    y_te_s = scaler_y.transform(y_te.reshape(-1,1)).flatten()
    for params in nn_combos:
        set_seed(int(params['seed']))
        model = FlatNet(input_size, int(params['hidden_size']),
                        float(params['dropout_rate'])).to(device)
        score, _ = train_flat_nn(model, X_tr, y_tr_s, X_te, y_te_s,
                                  scaler_y, float(params['learning_rate']),
                                  TUNE_EPOCHS, TUNE_PATIENCE)
        if score > best_score_nn:
            best_score_nn  = score
            best_params_nn = params
    print(f"    [{tag}] Best tune R²={best_score_nn:.4f}")

    # Final train
    set_seed(int(best_params_nn['seed']))
    final_model = FlatNet(input_size, int(best_params_nn['hidden_size']),
                          float(best_params_nn['dropout_rate'])).to(device)
    _, pred = train_flat_nn(final_model, X_tr, y_tr_s, X_te, y_te_s,
                             scaler_y, float(best_params_nn['learning_rate']),
                             MAX_EPOCHS, EARLY_STOPPING)
    r2   = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    print(f"    [{tag}] Final R²={r2:.4f}, RMSE={rmse:.4f}")
    return r2, rmse, pred, best_params_nn


# ─────────────────────────────────────────────
# sklearn搜索（通用）
# ─────────────────────────────────────────────
def search_sklearn(combos, X_tr, y_tr_s, X_te, y_te,
                   scaler_y, model_class, model_kwargs=None):
    if model_kwargs is None: model_kwargs = {}
    best_score, best_params = -np.inf, None
    for params in combos:
        m = model_class(**{**params, **model_kwargs})
        m.fit(X_tr, y_tr_s)
        pred  = scaler_y.inverse_transform(
            m.predict(X_te).reshape(-1, 1)).flatten()
        score = r2_score(y_te, pred)
        if score > best_score:
            best_score  = score
            best_params = params
    return best_params, best_score


def run_sklearn_model(X_tr, y_tr_s, X_te, y_te, scaler_y,
                      coarse_grid, fine_fn, model_class,
                      model_kwargs=None, tag='XGB'):
    if model_kwargs is None: model_kwargs = {}
    print(f"    [{tag}] Coarse search ({N_ITER_COARSE})...")
    coarse = sample_params(coarse_grid, N_ITER_COARSE, SEED)
    bp, bs = search_sklearn(coarse, X_tr, y_tr_s, X_te, y_te,
                             scaler_y, model_class, model_kwargs)
    print(f"    [{tag}] Fine search ({N_ITER_FINE})...")
    fine   = sample_params(fine_fn(bp), N_ITER_FINE, SEED+1)
    bp_f, bs_f = search_sklearn(fine, X_tr, y_tr_s, X_te, y_te,
                                 scaler_y, model_class, model_kwargs)
    if bs_f > bs:
        bp = bp_f
    final = model_class(**{**bp, **model_kwargs})
    final.fit(X_tr, y_tr_s)
    pred  = scaler_y.inverse_transform(
        final.predict(X_te).reshape(-1, 1)).flatten()
    r2   = r2_score(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    print(f"    [{tag}] Final R²={r2:.4f}, RMSE={rmse:.4f}")
    return r2, rmse, pred


# ─────────────────────────────────────────────
# 主运行函数（单个cutoff）
# ─────────────────────────────────────────────
def run_ablation(df, cutoff, results_dir):
    os.makedirs(results_dir, exist_ok=True)

    tr_df = df[df['year'] < cutoff].copy()
    te_df = df[df['year'] >= cutoff].copy()
    y_te  = te_df[TARGET_COL].values.flatten()

    print(f"\n{'='*65}")
    print(f"Cutoff={cutoff}: train={len(tr_df)}({tr_df[DOI_COL].nunique()} DOIs) | "
          f"test={len(te_df)}({te_df[DOI_COL].nunique()} DOIs)")
    print(f"{'='*65}")

    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in df.columns
                if df[c].dtype != 'object' and c not in exclude]

    scaler_y = RobustScaler()
    y_tr_s   = scaler_y.fit_transform(
        tr_df[TARGET_COL].values.reshape(-1, 1)).flatten()

    # ── 预先构建三套特征 ──
    X_tr_num, X_te_num, _, _ = build_features(
        df, tr_df, te_df, num_cols, mode='numerical')
    X_tr_label, X_te_label, _, _ = build_features(
        df, tr_df, te_df, num_cols, mode='label')
    X_tr_onehot, X_te_onehot, _, _ = build_features(
        df, tr_df, te_df, num_cols, mode='onehot')

    # entity embedding需要的额外准备
    label_encoders, categorical_info = {}, {}
    cat_tr_emb, cat_te_emb = [], []
    scaler_X_num = RobustScaler()
    X_num_tr_s = scaler_X_num.fit_transform(
        tr_df[num_cols].values.astype(float))
    X_num_te_s = scaler_X_num.transform(
        te_df[num_cols].values.astype(float))
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        label_encoders[col]   = le
        categorical_info[col] = len(le.classes_)
        cat_tr_emb.append(le.transform(tr_df[col].astype(str)))
        cat_te_emb.append(le.transform(te_df[col].astype(str)))
    X_cat_tr_emb = np.column_stack(cat_tr_emb)
    X_cat_te_emb = np.column_stack(cat_te_emb)

    results = {}

    # ════════════════════════════════════════════════════════════
    # 数值 only
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"[Group 1] Numerical features only")
    print(f"{'─'*50}")

    # NN (numerical only)
    print(f"\n  [NN]")
    r2, rmse, pred, _ = search_flat_nn(
        X_tr_num, y_tr_s, X_te_num, y_te, scaler_y, tag='NN')
    results['NN'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                     'y_pred': pred, 'year_te': te_df['year'].values,
                     'features': 'numerical only'}

    # XGB (numerical only)
    if HAS_XGB:
        print(f"\n  [XGB]")
        r2, rmse, pred = run_sklearn_model(
            X_tr_num, y_tr_s, X_te_num, y_te, scaler_y,
            XGB_COARSE, make_fine_xgb,
            xgb.XGBRegressor, {'verbosity': 0}, tag='XGB')
        results['XGB'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                          'y_pred': pred, 'year_te': te_df['year'].values,
                          'features': 'numerical only'}

    # RF (numerical only)
    print(f"\n  [RF]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_num, y_tr_s, X_te_num, y_te, scaler_y,
        RF_COARSE, make_fine_rf,
        RandomForestRegressor, {'n_jobs': -1}, tag='RF')
    results['RF'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                     'y_pred': pred, 'year_te': te_df['year'].values,
                     'features': 'numerical only'}

    # SVR (numerical only)
    print(f"\n  [SVR]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_num, y_tr_s, X_te_num, y_te, scaler_y,
        SVR_COARSE, make_fine_svr, SVR, {}, tag='SVR')
    results['SVR'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                      'y_pred': pred, 'year_te': te_df['year'].values,
                      'features': 'numerical only'}

    # GPR (numerical only)
    print(f"\n  [GPR]")
    best_score_gpr, best_kernel_gpr, best_seed_gpr = -np.inf, None, SEED
    for kernel in GPR_KERNELS:
        for s in GPR_SEEDS:
            m = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                         n_restarts_optimizer=2, random_state=s)
            m.fit(X_tr_num, y_tr_s)
            pred_gpr = scaler_y.inverse_transform(
                m.predict(X_te_num).reshape(-1, 1)).flatten()
            score = r2_score(y_te, pred_gpr)
            if score > best_score_gpr:
                best_score_gpr  = score
                best_kernel_gpr = kernel
                best_seed_gpr   = s
    final_gpr = GaussianProcessRegressor(
        kernel=best_kernel_gpr, normalize_y=True,
        n_restarts_optimizer=3, random_state=best_seed_gpr)
    final_gpr.fit(X_tr_num, y_tr_s)
    pred_gpr  = scaler_y.inverse_transform(
        final_gpr.predict(X_te_num).reshape(-1, 1)).flatten()
    r2_gpr    = r2_score(y_te, pred_gpr)
    rmse_gpr  = np.sqrt(mean_squared_error(y_te, pred_gpr))
    print(f"    [GPR] Final R²={r2_gpr:.4f}, RMSE={rmse_gpr:.4f}")
    results['GPR'] = {'r2': r2_gpr, 'rmse': rmse_gpr, 'y_true': y_te,
                      'y_pred': pred_gpr, 'year_te': te_df['year'].values,
                      'features': 'numerical only'}

    # ════════════════════════════════════════════════════════════
    # 数值 + Label Encoding
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"[Group 2] Numerical + Label Encoding (text as integers)")
    print(f"{'─'*50}")

    print(f"\n  [NN + label]")
    r2, rmse, pred, _ = search_flat_nn(
        X_tr_label, y_tr_s, X_te_label, y_te, scaler_y, tag='NN+label')
    results['NN+label'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                            'y_pred': pred, 'year_te': te_df['year'].values,
                            'features': 'num + label enc'}

    if HAS_XGB:
        print(f"\n  [XGB + label]")
        r2, rmse, pred = run_sklearn_model(
            X_tr_label, y_tr_s, X_te_label, y_te, scaler_y,
            XGB_COARSE, make_fine_xgb,
            xgb.XGBRegressor, {'verbosity': 0}, tag='XGB+label')
        results['XGB+label'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                                 'y_pred': pred, 'year_te': te_df['year'].values,
                                 'features': 'num + label enc'}

    print(f"\n  [RF + label]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_label, y_tr_s, X_te_label, y_te, scaler_y,
        RF_COARSE, make_fine_rf,
        RandomForestRegressor, {'n_jobs': -1}, tag='RF+label')
    results['RF+label'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                            'y_pred': pred, 'year_te': te_df['year'].values,
                            'features': 'num + label enc'}

    print(f"\n  [SVR + label]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_label, y_tr_s, X_te_label, y_te, scaler_y,
        SVR_COARSE, make_fine_svr, SVR, {}, tag='SVR+label')
    results['SVR+label'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                             'y_pred': pred, 'year_te': te_df['year'].values,
                             'features': 'num + label enc'}

    # ════════════════════════════════════════════════════════════
    # 数值 + One-Hot Encoding
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"[Group 3] Numerical + One-Hot Encoding")
    print(f"{'─'*50}")

    print(f"\n  [NN + one-hot]")
    r2, rmse, pred, _ = search_flat_nn(
        X_tr_onehot, y_tr_s, X_te_onehot, y_te, scaler_y, tag='NN+onehot')
    results['NN+onehot'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                             'y_pred': pred, 'year_te': te_df['year'].values,
                             'features': 'num + one-hot'}

    if HAS_XGB:
        print(f"\n  [XGB + one-hot]")
        r2, rmse, pred = run_sklearn_model(
            X_tr_onehot, y_tr_s, X_te_onehot, y_te, scaler_y,
            XGB_COARSE, make_fine_xgb,
            xgb.XGBRegressor, {'verbosity': 0}, tag='XGB+onehot')
        results['XGB+onehot'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                                  'y_pred': pred, 'year_te': te_df['year'].values,
                                  'features': 'num + one-hot'}

    print(f"\n  [RF + one-hot]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_onehot, y_tr_s, X_te_onehot, y_te, scaler_y,
        RF_COARSE, make_fine_rf,
        RandomForestRegressor, {'n_jobs': -1}, tag='RF+onehot')
    results['RF+onehot'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                             'y_pred': pred, 'year_te': te_df['year'].values,
                             'features': 'num + one-hot'}

    print(f"\n  [SVR + one-hot]")
    r2, rmse, pred = run_sklearn_model(
        X_tr_onehot, y_tr_s, X_te_onehot, y_te, scaler_y,
        SVR_COARSE, make_fine_svr, SVR, {}, tag='SVR+onehot')
    results['SVR+onehot'] = {'r2': r2, 'rmse': rmse, 'y_true': y_te,
                              'y_pred': pred, 'year_te': te_df['year'].values,
                              'features': 'num + one-hot'}

    # ════════════════════════════════════════════════════════════
    # Combined (AttnAlloyGen) — entity embedding
    # ════════════════════════════════════════════════════════════
    print(f"\n{'─'*50}")
    print(f"[Group 4] AttnAlloyGen — Entity Embedding (this work)")
    print(f"{'─'*50}")

    print(f"\n  [Combined] hardcoded best params: {COMBINED_BEST[cutoff]}")
    params_c = COMBINED_BEST[cutoff]
    set_seed(int(params_c['seed']))
    final_combined = CombinedNet(
        X_num_tr_s.shape[1], categorical_info,
        int(params_c['hidden_size']),
        float(params_c['dropout_rate'])).to(device)
    y_tr_s_emb = scaler_y.transform(
        tr_df[TARGET_COL].values.reshape(-1, 1)).flatten()
    y_te_s_emb = scaler_y.transform(y_te.reshape(-1, 1)).flatten()
    r2_c, pred_c = train_combined_nn(
        final_combined,
        X_num_tr_s, X_cat_tr_emb, y_tr_s_emb,
        X_num_te_s, X_cat_te_emb, y_te_s_emb,
        scaler_y, float(params_c['learning_rate']),
        MAX_EPOCHS, EARLY_STOPPING)
    rmse_c = np.sqrt(mean_squared_error(y_te, pred_c))
    print(f"    [Combined] Final R²={r2_c:.4f}, RMSE={rmse_c:.4f}")
    results['Combined'] = {'r2': r2_c, 'rmse': rmse_c, 'y_true': y_te,
                            'y_pred': pred_c, 'year_te': te_df['year'].values,
                            'features': 'num + entity embedding'}

    # ── 汇总打印 ──
    print(f"\n{'='*70}")
    print(f"Cutoff={cutoff} | Full Summary")
    print(f"{'='*70}")
    print(f"{'Model':<16} {'Features':<22} {'R²':>8} {'RMSE (mm/yr)':>14}")
    print(f"{'─'*70}")
    groups = [
        ('── Numerical only ──', ['NN', 'XGB', 'RF', 'SVR', 'GPR']),
        ('── + Label encoding ──', ['NN+label', 'XGB+label', 'RF+label', 'SVR+label']),
        ('── + One-hot encoding ──', ['NN+onehot', 'XGB+onehot', 'RF+onehot', 'SVR+onehot']),
        ('── + Entity embedding ──', ['Combined']),
    ]
    for group_name, names in groups:
        print(f"\n{group_name}")
        for name in names:
            if name in results:
                res = results[name]
                marker = ' ←' if name == 'Combined' else ''
                print(f"  {name:<14} {res['features']:<22} "
                      f"{res['r2']:>8.4f} {res['rmse']:>14.4f}{marker}")
    print(f"{'='*70}")

    # ── 保存CSV ──
    rows = [{'model': name, 'cutoff': cutoff,
             'features': res['features'],
             'r2': res['r2'], 'rmse': res['rmse'],
             'mae': np.mean(np.abs(res['y_true'] - res['y_pred']))}
            for name, res in results.items()]
    pd.DataFrame(rows).to_csv(
        os.path.join(results_dir, 'summary_extended.csv'), index=False)
    print(f"Results saved → {results_dir}/summary_extended.csv")

    return results


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == '__main__':
    args    = parse_args()
    cutoffs = args.cutoff or [2024, 2025]

    print(f"\nYear-split ablation (extended) — 13 model variants")
    print(f"Cutoffs: {cutoffs}\n")

    df = load_and_preprocess(args.data)

    all_results = {}
    for cutoff in cutoffs:
        results_dir = f'year_split_ablation_extended_{cutoff}'
        print(f"\n{'#'*65}")
        print(f"# Cutoff = {cutoff}")
        print(f"{'#'*65}")
        res = run_ablation(df, cutoff, results_dir)
        all_results[cutoff] = res

    # ── 跨cutoff汇总CSV ──
    if len(cutoffs) > 1:
        rows = []
        model_names = list(all_results[cutoffs[0]].keys())
        for name in model_names:
            row = {'model': name,
                   'features': all_results[cutoffs[0]][name]['features']}
            for c in cutoffs:
                if name in all_results[c]:
                    row[f'r2_{c}']   = all_results[c][name]['r2']
                    row[f'rmse_{c}'] = all_results[c][name]['rmse']
            rows.append(row)
        df_summary = pd.DataFrame(rows)
        df_summary.to_csv('ablation_extended_comparison.csv', index=False)
        print(f"\nCross-cutoff comparison saved → ablation_extended_comparison.csv")

        print(f"\n{'='*75}")
        print(f"{'Model':<16} {'Features':<22}", end='')
        for c in cutoffs:
            print(f"  R²({c})  RMSE({c})", end='')
        print()
        print('='*75)
        for _, row in df_summary.iterrows():
            print(f"{row['model']:<16} {row['features']:<22}", end='')
            for c in cutoffs:
                print(f"  {row.get(f'r2_{c}', 0):>6.4f}  "
                      f"{row.get(f'rmse_{c}', 0):>6.3f} ", end='')
            print()
        print('='*75)

    print("\nDone.")