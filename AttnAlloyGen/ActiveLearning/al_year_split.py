import os, re, pickle, random, warnings, argparse, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import RobustScaler, LabelEncoder
from itertools import product as iterproduct
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════════
# 断点续跑
# ══════════════════════════════════════════════════════════════════════════════
_PROGRESS_PATH = None

def load_progress():
    if _PROGRESS_PATH and os.path.exists(_PROGRESS_PATH):
        with open(_PROGRESS_PATH) as f:
            return json.load(f)
    return {}

def save_progress(progress):
    if _PROGRESS_PATH:
        with open(_PROGRESS_PATH, 'w') as f:
            json.dump(progress, f, indent=2)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
os.environ['PYTHONHASHSEED'] = '42'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


# ══════════════════════════════════════════════════════════════════════════════
# 所有固定参数（含 3rd / 4th，不再搜索）
# ══════════════════════════════════════════════════════════════════════════════
MAX_EPOCHS     = 1500
EARLY_STOPPING = 300
SEED = 456

BASELINE_PARAMS = {
    2024: {'hidden_size': 112, 'dropout_rate': 0.06,  'learning_rate': 0.0008, 'seed': 143750},
    2025: {'hidden_size': 112, 'dropout_rate': 0.12,  'learning_rate': 0.0006, 'seed': 119700},
}

FIXED_ITER_PARAMS = {
    '1st Iteration_cutoff2025': {'hidden_size': 112, 'dropout_rate': 0.09, 'learning_rate': 0.0004, 'seed': 119700},
    '2nd Iteration_cutoff2025': {'hidden_size': 112, 'dropout_rate': 0.09, 'learning_rate': 0.002,  'seed': 169700},
    '3rd Iteration_cutoff2025': {'hidden_size': 112, 'dropout_rate': 0.09, 'learning_rate': 0.0006, 'seed': 120650},
    '4th Iteration_cutoff2025': {'hidden_size': 112, 'dropout_rate': 0.03, 'learning_rate': 0.0004, 'seed': 120650},
}


# ══════════════════════════════════════════════════════════════════════════════
# 数据配置
# ══════════════════════════════════════════════════════════════════════════════
EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0',
    '10.1007/s11665-026-13708-2',
    '10.1016/j.jma.2018.11.004',
    '10.1016/j.jmrt.2022.05.161',
}

DOI_YEAR_MAP = {
    '10.1002/maco.201609141':           2016,
    '10.1002/sia.7075':                 2022,
    '10.1007/s11661-021-06548-3':       2021,
    '10.1007/s11665-020-05034-y':       2020,
    '10.1007/s11665-022-07081-z':       2022,
    '10.1007/s11665-023-08526-9':       2023,
    '10.1007/s11665-025-13079-0':       2025,
    '10.1007/s11665-026-13543-5':       2026,
    '10.1007/s11665-026-13708-2':       2026,
    '10.1007/s40195-022-01451-1':       2022,
    '10.1016/ j.jma.2022.12.005':       2022,
    '10.1016/S1005-0302(12)60186-8':    2012,
    '10.1016/j.apsusc.2016.01.283':     2016,
    '10.1016/j.apsusc.2020.147761':     2020,
    '10.1016/j.corcom.2022.10.002':     2022,
    '10.1016/j.corsci.2011.10.033':     2011,
    '10.1016/j.corsci.2017.01.022':     2017,
    '10.1016/j.corsci.2017.02.009':     2017,
    '10.1016/j.corsci.2020.108451':     2020,
    '10.1016/j.corsci.2020.108487':     2020,
    '10.1016/j.corsci.2020.108831':     2020,
    '10.1016/j.corsci.2020.108958':     2020,
    '10.1016/j.corsci.2020.108998':     2020,
    '10.1016/j.corsci.2021.110023':     2021,
    '10.1016/j.corsci.2022.110861':     2022,
    '10.1016/j.corsci.2023.111021':     2023,
    '10.1016/j.corsci.2024.112180':     2024,
    '10.1016/j.electacta.2025.146957':  2025,
    '10.1016/j.intermet.2023.108000':   2023,
    '10.1016/j.jallcom.2014.10.177':    2014,
    '10.1016/j.jallcom.2020.153839':    2020,
    '10.1016/j.jallcom.2020.154344':    2020,
    '10.1016/j.jallcom.2020.156619':    2020,
    '10.1016/j.jallcom.2024.174068':    2024,
    '10.1016/j.jallcom.2025.180504':    2025,
    '10.1016/j.jallcom.2026.187172':    2026,
    '10.1016/j.jma.2016.12.002':        2016,
    '10.1016/j.jma.2018.09.002':        2018,
    '10.1016/j.jma.2018.11.004':        2018,
    '10.1016/j.jma.2020.03.014':        2020,
    '10.1016/j.jma.2020.06.004':        2020,
    '10.1016/j.jma.2022.08.003':        2022,
    '10.1016/j.jma.2026.102035':        2026,
    '10.1016/j.jmrt.2022.05.161':       2022,
    '10.1016/j.jmrt.2025.04.024':       2025,
    '10.1016/j.jmst.2020.05.067':       2020,
    '10.1016/j.jmst.2023.01.058':       2023,
    '10.1016/j.jmst.2023.08.068':       2023,
    '10.1016/j.jmst.2025.01.020':       2025,
    '10.1016/j.jpcs.2021.109952':       2021,
    '10.1016/j.matchar.2021.111294':    2021,
    '10.1016/j.matchar.2022.112121':    2022,
    '10.1016/j.matdes.2022.110980':     2022,
    '10.1016/j.mtcomm.2024.111277':     2024,
    '10.1016/j.scriptamat.2018.11.046': 2018,
    '10.1038/s41529-026-00755-2':       2026,
    '10.1088/2053-1591/ab51ab':         2019,
    '10.11900/0412.1961.2024.00230':    2024,
    '10.20964/2017.11.14':              2017,
}

CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]

NUMERICAL_START_COL = 0
NUMERICAL_END_COL   = 29
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'


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
        return '+'.join([f"{float(t):.0f}°C" for t in m])
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

def load_and_preprocess(path):
    df = pd.read_excel(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try:    df[col] = df[col].fillna(df[col].median())
            except: mode_val = df[col].mode(); df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
    df = df[df[TARGET_COL] <= 30]
    mean_ = df[TARGET_COL].mean(); std_ = df[TARGET_COL].std()
    df = df[df[TARGET_COL] <= mean_ + 2.5 * std_]
    df = df[df['Test method'] == 'pw'].copy()
    print("Text standardization:")
    df = apply_text_standardization(df)
    num_cols = df.columns[NUMERICAL_START_COL:NUMERICAL_END_COL].tolist()
    num_cols = [c for c in num_cols if df[c].dtype != 'object' and c not in [TARGET_COL, DOI_COL]]
    zero_cols = [c for c in num_cols if (df[c] == 0).all()]
    if zero_cols:
        print(f"Dropping zero columns: {zero_cols}")
        df = df.drop(columns=zero_cols)
    before = len(df)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    n_excl = before - len(df)
    if n_excl > 0:
        print(f"Excluded {len(EXCLUDED_DOIS)} DOIs ({n_excl} samples)")
    df = df.reset_index(drop=True)
    print(f"\nData (pw only, after exclusion): {len(df)} samples | {df[DOI_COL].nunique()} unique DOIs")
    cr = df[TARGET_COL]
    print(f"CR stats: mean={cr.mean():.2f}, std={cr.std():.2f}, min={cr.min():.3f}, max={cr.max():.2f} mm/yr")
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    unknown = df[df['year'].isna()][DOI_COL].unique()
    if len(unknown) > 0:
        print(f"\n[警告] 以下DOI无年份映射，将被排除:")
        for d in unknown: print(f"  {d}")
        df = df[df['year'].notna()].copy()
    return df

def load_iteration_data(path):
    df = pd.read_excel(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'unknown')
        else:
            try:    df[col] = df[col].fillna(df[col].median())
            except: df[col] = df[col].fillna(df[col].mode()[0])
    df = apply_text_standardization(df)
    df = df.drop(columns=[c for c in [DOI_COL, 'year'] if c in df.columns], errors='ignore')
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 模型
# ══════════════════════════════════════════════════════════════════════════════
class CombinedNet(nn.Module):
    def __init__(self, numerical_size, categorical_info, hidden_size=112, dropout_rate=0.12):
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
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x_num, x_cat):
        emb = [self.embeddings[fn](x_cat[:, i]) for i, fn in enumerate(self.embeddings)]
        x = torch.cat([x_num] + emb, dim=1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        return self.output(x)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_nn(model, X_tr_t, y_tr_t, X_te_t, y_te_t,
             scaler_y, lr, max_epochs, patience, X_cat_tr_t, X_cat_te_t, device):
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
    r2   = r2_score(scaler_y.inverse_transform(y_te_t.cpu().numpy()).flatten(), pred)
    return r2, pred, best_state

def encode_split(tr_df, te_df, full_df, device):
    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in full_df.columns
                if full_df[c].dtype != 'object' and c not in exclude]
    X_num_tr = tr_df[num_cols].values.astype(float)
    X_num_te = te_df[num_cols].values.astype(float)
    y_tr_raw = tr_df[TARGET_COL].values.reshape(-1, 1)
    y_te_raw = te_df[TARGET_COL].values.reshape(-1, 1)
    scaler_X = RobustScaler(); scaler_y = RobustScaler()
    X_num_tr_s = scaler_X.fit_transform(X_num_tr)
    X_num_te_s = scaler_X.transform(X_num_te)
    y_tr_s     = scaler_y.fit_transform(y_tr_raw).flatten()
    y_te_s     = scaler_y.transform(y_te_raw).flatten()
    label_encoders, categorical_info = {}, {}
    cat_tr, cat_te = [], []
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(full_df[col].astype(str))
        label_encoders[col]   = le
        categorical_info[col] = len(le.classes_)
        cat_tr.append(le.transform(tr_df[col].astype(str)))
        cat_te.append(le.transform(te_df[col].astype(str)))
    X_tr_t     = torch.FloatTensor(X_num_tr_s).to(device)
    X_te_t     = torch.FloatTensor(X_num_te_s).to(device)
    X_cat_tr_t = torch.LongTensor(np.column_stack(cat_tr)).to(device)
    X_cat_te_t = torch.LongTensor(np.column_stack(cat_te)).to(device)
    y_tr_t     = torch.FloatTensor(y_tr_s.reshape(-1, 1)).to(device)
    y_te_t     = torch.FloatTensor(y_te_s.reshape(-1, 1)).to(device)
    return {
        'X_tr_t': X_tr_t, 'X_te_t': X_te_t,
        'X_cat_tr_t': X_cat_tr_t, 'X_cat_te_t': X_cat_te_t,
        'y_tr_t': y_tr_t, 'y_te_t': y_te_t,
        'y_te_orig': y_te_raw.flatten(),
        'scaler_y': scaler_y, 'scaler_X': scaler_X,
        'categorical_info': categorical_info,
        'label_encoders': label_encoders,
        'num_cols': num_cols,
        'input_size': X_num_tr_s.shape[1],
    }

def quick_train(enc, params, max_epochs, patience, device):
    set_seed(int(params['seed']))
    model = CombinedNet(enc['input_size'], enc['categorical_info'],
                        hidden_size=int(params['hidden_size']),
                        dropout_rate=float(params['dropout_rate'])).to(device)
    r2, pred, state = train_nn(
        model, enc['X_tr_t'], enc['y_tr_t'],
        enc['X_te_t'], enc['y_te_t'],
        enc['scaler_y'], float(params['learning_rate']),
        max_epochs, patience,
        enc['X_cat_tr_t'], enc['X_cat_te_t'], device)
    return r2, pred, state


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def run_one_cutoff(base_df, cutoff, iteration_list, log_base, model_base, device):
    print(f"\n{'█'*70}")
    print(f"  CUTOFF = {cutoff}")
    print(f"{'█'*70}")

    model_dir = os.path.join(model_base, f'cutoff{cutoff}')
    os.makedirs(model_dir, exist_ok=True)

    train_base = base_df[base_df['year'] < cutoff].copy().reset_index(drop=True)
    test_fixed = base_df[base_df['year'] >= cutoff].copy().reset_index(drop=True)
    print(f"  Train base: {len(train_base)} | Test fixed: {len(test_fixed)}")

    results, params_record = [], {}

    # Baseline
    print(f"\n{'='*60}\n  [Baseline] cutoff={cutoff}")
    full_df = pd.concat([train_base, test_fixed], ignore_index=True)
    enc = encode_split(train_base, test_fixed, full_df, device)
    bl_params = BASELINE_PARAMS[cutoff]
    r2, pred, _ = quick_train(enc, bl_params, MAX_EPOCHS, EARLY_STOPPING, device)
    rmse = np.sqrt(mean_squared_error(enc['y_te_orig'], pred))
    print(f"  Baseline: R²={r2:.4f}, RMSE={rmse:.4f}")
    print(f"  （预期 cutoff=2025: R²≈0.809；cutoff=2024: R²≈0.628）")
    results.append({'name': 'Baseline', 'r2': r2, 'rmse': rmse,
                    'pred': pred, 'true': enc['y_te_orig'],
                    'params': bl_params, 'n_train': len(train_base)})
    params_record['Baseline'] = bl_params

    current_train = train_base.copy()

    for iter_path, iter_name in iteration_list:
        print(f"\n{'='*60}\n  [{iter_name}] cutoff={cutoff}  ←  {iter_path}")
        new_data = load_iteration_data(iter_path)
        current_train = pd.concat([current_train, new_data], ignore_index=True)
        print(f"  累积训练集: {len(current_train)} 条")

        full_df_iter = pd.concat([current_train, test_fixed], ignore_index=True)
        enc = encode_split(current_train, test_fixed, full_df_iter, device)

        iter_key   = f"{iter_name}_cutoff{cutoff}"
        model_path = os.path.join(model_dir, f"{iter_name.replace(' ','_').lower()}_model.pth")

        params = FIXED_ITER_PARAMS.get(iter_key)
        if params is None:
            raise KeyError(f"找不到固定参数: {iter_key}，请在 FIXED_ITER_PARAMS 中添加")
        print(f"  [固定参数] {params}")

        r2, pred, state = quick_train(enc, params, MAX_EPOCHS, EARLY_STOPPING, device)
        rmse = np.sqrt(mean_squared_error(enc['y_te_orig'], pred))
        torch.save(state, model_path)
        print(f"  结果: R²={r2:.4f}, RMSE={rmse:.4f}")

        results.append({'name': iter_name, 'r2': r2, 'rmse': rmse,
                        'pred': pred, 'true': enc['y_te_orig'],
                        'params': params, 'n_train': len(current_train)})
        params_record[iter_key] = params

    return results, params_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', nargs='*', type=int, default=[2025])
    parser.add_argument('--data',   type=str,  default='database_colab_text_new1.xlsx')
    parser.add_argument('--iter1',  type=str,  default='1st_iteration.xlsx')
    parser.add_argument('--iter2',  type=str,  default='2nd_iteration.xlsx')
    parser.add_argument('--iter3',  type=str,  default='3rd_iteration.xlsx')
    parser.add_argument('--iter4',  type=str,  default='4th_iteration.xlsx')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    set_seed(SEED)

    MODEL_BASE = 'al_year_split_models'
    os.makedirs(MODEL_BASE, exist_ok=True)

    print("\n[加载数据]")
    base_df = load_and_preprocess(args.data)

    iteration_list = [
        (args.iter1, '1st Iteration'),
        (args.iter2, '2nd Iteration'),
        (args.iter3, '3rd Iteration'),
        (args.iter4, '4th Iteration'),
    ]

    all_results, all_params = {}, {}
    for cutoff in args.cutoff:
        results, params_rec = run_one_cutoff(
            base_df, cutoff, iteration_list, 'al_year_split_logs', MODEL_BASE, device)
        all_results[cutoff] = results
        all_params.update(params_rec)

    for cutoff, results in all_results.items():
        print(f"\n{'='*70}\n汇总 cutoff={cutoff}\n{'='*70}")
        print(f"{'Model':<18} {'N_train':>8} {'Test R²':>9} {'RMSE':>8}  参数")
        print(f"{'-'*75}")
        for r in results:
            p = r['params']
            print(f"{r['name']:<18} {r['n_train']:>8} {r['r2']:>9.4f} {r['rmse']:>8.4f}  "
                  f"seed={p['seed']}, h={p['hidden_size']}, "
                  f"dr={p['dropout_rate']}, lr={p['learning_rate']}")

    plot_all(all_results)
    print("\n✅ 完成！")


def plot_all(all_results):
    cutoffs = list(all_results.keys())
    n_cols  = max(len(v) for v in all_results.values())
    n_rows  = len(cutoffs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), squeeze=False)
    for row, cutoff in enumerate(cutoffs):
        for col, r in enumerate(all_results[cutoff]):
            ax = axes[row][col]
            pred, true = r['pred'], r['true']
            vmin = max(0, min(true.min(), pred.min()) * 0.95)
            vmax = max(true.max(), pred.max()) * 1.05
            ax.scatter(pred, true, c='#FFA726', s=35, alpha=0.85, zorder=3)
            ax.plot([vmin, vmax], [vmin, vmax], 'k-', lw=1.2, alpha=0.7)
            ax.set_xlim(vmin, vmax); ax.set_ylim(vmin, vmax)
            ax.set_aspect('equal', adjustable='box')
            ax.text(0.05, 0.95,
                    f"Test R²: {r['r2']:.3f}\nRMSE: {r['rmse']:.3f} mm/yr",
                    transform=ax.transAxes, fontsize=9, fontweight='bold', va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='k', alpha=0.9))
            ax.set_xlabel('Predicted CR (mm/yr)')
            ax.set_ylabel('Experimental CR (mm/yr)')
            ax.set_title(f"cutoff={cutoff} | {r['name']}", fontweight='bold', loc='left')
            ax.text(0.98, 0.05, f"n={r['n_train']}",
                    transform=ax.transAxes, ha='right', fontsize=8, color='gray')
        for col in range(len(all_results[cutoff]), n_cols):
            axes[row][col].set_visible(False)
    plt.tight_layout()
    plt.savefig('al_year_split_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("散点图: al_year_split_scatter.png")


if __name__ == '__main__':
    main()