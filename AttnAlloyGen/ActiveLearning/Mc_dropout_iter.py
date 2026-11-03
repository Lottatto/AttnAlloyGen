"""
mc_dropout_iter.py
==================
基于 al_analysis.py 的结构，对每轮 iteration 选取的成分
用 MC Dropout 计算预测均值和不确定度 σ。

直接放在和 al_analysis.py 同一目录下运行：
    python mc_dropout_iter.py

输出：
    mc_dropout_results.xlsx  — 每条成分的预测均值、σ、实验值
    控制台打印各轮σ统计
"""

import os, re, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder

# ══════════════════════════════════════════════════════════════════════════════
# 路径配置（和 al_analysis.py 保持一致）
# ══════════════════════════════════════════════════════════════════════════════
BASE_DATA_PATH   = 'database_colab_text_new1.xlsx'
MODEL_BASE_DIR   = 'al_year_split_models'
BEST_PARAMS_PATH = 'al_best_params.json'
CUTOFF           = 2025
N_PASSES         = 30  # MC Dropout 前向传播次数

BASELINE_MODEL  = r'D:\vscode\corrosion rate\revision\generation\year_split_results_2025\model_cutoff2025.pth'
BASELINE_PARAMS = {'hidden_size': 112, 'dropout_rate': 0.12,
                   'learning_rate': 0.0006, 'seed': 119700}

ITERATION_FILES = {
    '1st Iteration': '1st_iteration.xlsx',
    '2nd Iteration': '2nd_iteration.xlsx',
    '3rd Iteration': '3rd_iteration.xlsx',
    '4th Iteration': '4th_iteration.xlsx',
    '5th Iteration': '5th_iteration.xlsx',
}

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
# 文本标准化（从 al_analysis.py 复制）
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
# 编码（和 al_analysis.py 一致，用累积训练集 fit scaler）
# ══════════════════════════════════════════════════════════════════════════════
def build_encoders(train_df, iter_df, full_df):
    """
    用累积训练集 fit scaler 和 label encoder，
    返回对 iter_df 编码后的 tensor 和 scaler_y。
    """
    exclude  = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    num_cols = [c for c in full_df.columns
                if full_df[c].dtype != 'object' and c not in exclude]

    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(train_df[num_cols].values.astype(float))
    scaler_y.fit(train_df[TARGET_COL].values.reshape(-1, 1))

    # 对 iter_df 编码数值特征
    X_iter_s = scaler_X.transform(iter_df[num_cols].values.astype(float))

    # label encoder 用 full_df fit（和 al_analysis.py 一致）
    cat_info, les = {}, {}
    cat_iter = []
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        le.fit(full_df[col].astype(str))
        cat_info[col] = len(le.classes_)
        les[col] = le
        # 处理 iter_df 中可能出现训练集里没有的类别，映射到最近已知类别
        iter_vals = iter_df[col].astype(str)
        known = set(le.classes_)
        iter_vals = iter_vals.apply(lambda v: v if v in known else le.classes_[0])
        cat_iter.append(le.transform(iter_vals))

    return (X_iter_s,
            np.column_stack(cat_iter),
            scaler_y,
            cat_info,
            len(num_cols))


# ══════════════════════════════════════════════════════════════════════════════
# MC Dropout 预测
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
        for _ in range(n_passes):
            out = model(X_num_t, X_cat_t).cpu().numpy()
            preds.append(scaler_y.inverse_transform(out).flatten())

    preds = np.array(preds)  # (n_passes, n_samples)
    mean  = np.maximum(preds.mean(axis=0), 0.0)
    std   = preds.std(axis=0)
    return mean, std


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open(BEST_PARAMS_PATH) as f:
        best_params = json.load(f)

    print("\n[加载基础数据]")
    base_df    = load_and_preprocess(BASE_DATA_PATH)
    train_base = base_df[base_df['year'] < CUTOFF].copy().reset_index(drop=True)
    test_fixed = base_df[base_df['year'] >= CUTOFF].copy().reset_index(drop=True)
    print(f"Train base: {len(train_base)} | Test: {len(test_fixed)}")

    all_results = []

    # 每轮iteration对应的模型 = 该轮之前的训练集训练的模型
    # 同时需要用该模型对应的训练集来fit encoder（保证embedding维度一致）
    # train_snapshots[i] = 预测第i轮时模型所见过的训练集
    iter_names = list(ITERATION_FILES.keys())
    iter_files = list(ITERATION_FILES.values())

    # 构建每轮的训练集快照
    train_snapshots = []
    current = train_base.copy()
    train_snapshots.append(current.copy())  # baseline模型的训练集
    for iter_file in iter_files[:-1]:       # 1st/2nd/3rd iteration后的训练集
        if os.path.exists(iter_file):
            new_data = load_iteration_data(iter_file)
            current = pd.concat([current, new_data], ignore_index=True)
        train_snapshots.append(current.copy())

    # 模型路径和参数
    model_configs = []
    # 1st iteration用baseline模型
    model_configs.append((BASELINE_MODEL, BASELINE_PARAMS))
    # 2nd/3rd/4th用前一轮训练的模型
    for prev_name in ['1st Iteration', '2nd Iteration', '3rd Iteration', '4th Iteration']:
        mp = os.path.join(MODEL_BASE_DIR, f'cutoff{CUTOFF}',
                          f"{prev_name.replace(' ','_').lower()}_model.pth")
        pk = f"{prev_name}_cutoff{CUTOFF}"
        params = best_params.get(pk)
        model_configs.append((mp, params))

    for idx, (iter_name, iter_file) in enumerate(ITERATION_FILES.items()):
        print(f"\n{'='*50}\n[{iter_name}]")

        if not os.path.exists(iter_file):
            print(f"  [跳过] 找不到 {iter_file}")
            continue

        iter_df = load_iteration_data(iter_file)
        print(f"  成分数: {len(iter_df)}")

        model_path, params = model_configs[idx]
        if params is None:
            print(f"  [跳过] 找不到对应参数")
            continue
        if not os.path.exists(model_path):
            print(f"  [跳过] 找不到模型文件: {model_path}")
            continue

        # 关键：用该模型对应的训练集fit encoder，保证embedding维度一致
        train_for_this_model = train_snapshots[idx]
        print(f"  对应训练集大小: {len(train_for_this_model)}")
        full_df = pd.concat([train_for_this_model, test_fixed], ignore_index=True)
        X_num_s, X_cat, scaler_y, cat_info, input_size = build_encoders(
            train_for_this_model, iter_df, full_df)

        # MC Dropout
        mean_cr, std_cr = mc_dropout_predict(
            model_path, params, X_num_s, X_cat,
            scaler_y, cat_info, input_size, device, N_PASSES)

        exp_cr = iter_df[TARGET_COL].values if TARGET_COL in iter_df.columns \
                 else [None] * len(iter_df)

        for i in range(len(iter_df)):
            row = {
                'Iteration':                iter_name,
                'Sample':                   f'Sample_{i+1}',
                'Deformation Process':      iter_df['Deformation Process'].iloc[i],
                'Heat Treatment Process':   iter_df['Heat Treatment Process'].iloc[i],
                'Experimental CR (mm/yr)':  float(exp_cr[i]) if exp_cr[i] is not None else None,
                'Predicted Mean μ (mm/yr)': round(float(mean_cr[i]), 4),
                'Predicted Std σ (mm/yr)':  round(float(std_cr[i]), 4),
            }
            if row['Experimental CR (mm/yr)'] is not None:
                row['|Pred - Exp| (mm/yr)'] = round(
                    abs(row['Predicted Mean μ (mm/yr)'] - row['Experimental CR (mm/yr)']), 4)
            all_results.append(row)

        print(f"  预测均值μ: {[round(v,4) for v in mean_cr]}")
        print(f"  不确定度σ: {[round(v,4) for v in std_cr]}")
        print(f"  实验值:    {[round(float(v),4) if v is not None else None for v in exp_cr]}")

    # 汇总输出
    result_df = pd.DataFrame(all_results)
    out_path = 'mc_dropout_results.xlsx'
    result_df.to_excel(out_path, index=False)

    print(f"\n\n{'='*60}")
    print("各轮σ统计（用于回复信）：")
    for iter_name in ITERATION_FILES.keys():
        sub = result_df[result_df['Iteration'] == iter_name]
        if len(sub) == 0:
            continue
        print(f"  {iter_name}: "
              f"mean σ = {sub['Predicted Std σ (mm/yr)'].mean():.4f}, "
              f"max σ = {sub['Predicted Std σ (mm/yr)'].max():.4f} mm/yr")

    print(f"\n完整结果：")
    print(result_df.to_string(index=False))
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()