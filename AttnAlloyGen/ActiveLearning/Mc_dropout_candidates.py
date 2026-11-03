"""
mc_dropout_candidates_v2.py
============================
对候选池（cr_prediction_results.xlsx）中所有成分用 baseline 模型跑 MC Dropout。
编码逻辑与 Mc_dropout_iter.py 完全一致：
  - scaler 只用训练集 fit
  - label encoder 用 full_df（训练集+测试集）fit，候选池中未知类别映射到 le.classes_[0]

用法：
    python mc_dropout_candidates_v2.py

输出：
    mc_dropout_candidates_v2_results.xlsx
"""

import os, re, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# 路径配置 — 按你的实际路径修改
# ══════════════════════════════════════════════════════════════════════════════
BASE_DATA_PATH  = 'database_colab_text_new1.xlsx'       # 原始训练数据
CANDIDATE_PATH  = 'cr_prediction_results.xlsx'          # 候选池文件
OUTPUT_PATH     = 'mc_dropout_candidates_v2_results.xlsx'

BASELINE_MODEL  = r'D:\vscode\corrosion rate\revision\generation\year_split_results_2025\model_cutoff2025.pth'
BASELINE_PARAMS = {'hidden_size': 112, 'dropout_rate': 0.12,
                   'learning_rate': 0.0006, 'seed': 119700}

CUTOFF   = 2025
N_PASSES = 30

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


# ══════════════════════════════════════════════════════════════════════════════
# 文本标准化（与 Mc_dropout_iter.py 完全一致）
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
    """加载原始训练数据"""
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
# 编码（与 Mc_dropout_iter.py build_encoders 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
def build_encoders(train_df, cand_df, full_df):
    """
    - scaler 只用训练集 fit
    - label encoder 用 full_df（训练集+测试集）fit，与 Mc_dropout_iter.py 一致
    - 候选池中未知类别映射到 le.classes_[0]
    """
    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in full_df.columns
                if full_df[c].dtype != 'object' and c not in exclude]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(train_df[num_cols].values.astype(float))
    scaler_y.fit(train_df[TARGET_COL].values.reshape(-1, 1))

    # 候选池数值特征编码
    X_cand_s = scaler_X.transform(cand_df[num_cols].values.astype(float))

    # label encoder 用 full_df fit（和训练时完全一致）
    cat_info = {}
    cat_cand = []
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(full_df[col].astype(str))          # 只用full_df，不加候选池
        cat_info[col] = len(le.classes_)
        known = set(le.classes_)
        # 未知类别映射到 le.classes_[0]
        vals = cand_df[col].astype(str).apply(
            lambda v: v if v in known else le.classes_[0])
        cat_cand.append(le.transform(vals))

    return (X_cand_s,
            np.column_stack(cat_cand),
            scaler_y,
            cat_info,
            len(num_cols))


# ══════════════════════════════════════════════════════════════════════════════
# MC Dropout 预测（与 Mc_dropout_iter.py 完全一致）
# ══════════════════════════════════════════════════════════════════════════════
def mc_dropout_predict(model_path, params, X_num_s, X_cat,
                       scaler_y, cat_info, input_size, device, n_passes=30):
    model = CombinedNet(input_size, cat_info,
                        int(params['hidden_size']),
                        float(params['dropout_rate'])).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,
                                     weights_only=True))
    model.train()  # 保持 dropout 激活

    X_num_t = torch.FloatTensor(X_num_s).to(device)
    X_cat_t = torch.LongTensor(X_cat).to(device)

    preds = []
    with torch.no_grad():
        for i in range(n_passes):
            out = model(X_num_t, X_cat_t).cpu().numpy()
            preds.append(scaler_y.inverse_transform(out).flatten())
            if (i + 1) % 10 == 0:
                print(f"  Pass {i+1}/{n_passes}")

    preds   = np.array(preds)
    mean_cr = np.maximum(preds.mean(axis=0), 0.0)
    std_cr  = preds.std(axis=0)
    return mean_cr, std_cr


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. 加载训练数据
    print("\n[加载训练数据]")
    base_df   = load_and_preprocess(BASE_DATA_PATH)
    train_df  = base_df[base_df['year'] < CUTOFF].copy().reset_index(drop=True)
    test_df   = base_df[base_df['year'] >= CUTOFF].copy().reset_index(drop=True)
    full_df   = base_df.copy()   # 训练集+测试集，用于fit label encoder
    print(f"训练集: {len(train_df)} | 测试集: {len(test_df)}")

    # 2. 加载候选池，只做文本标准化
    print("\n[加载候选池]")
    cand_df = pd.read_excel(CANDIDATE_PATH)
    cand_df = apply_text_standardization(cand_df)
    print(f"候选池总数: {len(cand_df)}")

    # 3. 编码（与 Mc_dropout_iter.py 完全一致）
    print("\n[编码]")
    X_num_s, X_cat, scaler_y, cat_info, input_size = build_encoders(
        train_df, cand_df, full_df)
    print(f"数值特征维度: {input_size}")
    print(f"类别embedding信息: { {k: v for k, v in cat_info.items()} }")

    # 4. 加载模型并跑 MC Dropout
    print(f"\n[MC Dropout] 模型: {BASELINE_MODEL}")
    mean_cr, std_cr = mc_dropout_predict(
        BASELINE_MODEL, BASELINE_PARAMS,
        X_num_s, X_cat, scaler_y, cat_info, input_size,
        device, N_PASSES)

    # 5. 输出结果
    result_df = cand_df[['Rank', 'alloy_name', 'source',
                          'Predicted CR (mm/y)']].copy()
    result_df['Predicted Mean μ (mm/yr)'] = np.round(mean_cr, 4)
    result_df['Predicted Std σ (mm/yr)']  = np.round(std_cr, 4)

    result_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n结果已保存: {OUTPUT_PATH}")

    # 统计
    print(f"\n整体σ统计:")
    print(f"  mean σ = {std_cr.mean():.4f} mm/yr")
    print(f"  min σ  = {std_cr.min():.4f} mm/yr")
    print(f"  max σ  = {std_cr.max():.4f} mm/yr")

    # 对比1st iteration两个成分（Rank 107和268）
    iter1 = result_df[result_df['source'] == '1st_iteration']
    if len(iter1) > 0:
        print(f"\n1st iteration成分预测结果（与Mc_dropout_iter.py对比）:")
        print(iter1[['Rank', 'alloy_name', 'Predicted Mean μ (mm/yr)',
                      'Predicted Std σ (mm/yr)']].to_string(index=False))


if __name__ == '__main__':
    main()