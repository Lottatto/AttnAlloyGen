"""
cr_prediction.py
----------------
读取 candidates_WAEMCMC_full.csv（已包含候选池+iteration实验成分，
特征已补全），用 year_split 初始模型预测腐蚀速率，
按预测 CR 从低到高排序，iteration 实验成分行加粗，输出 xlsx。

用法：python cr_prediction.py
"""

import os, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl import load_workbook
from openpyxl.styles import Font

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR   = os.path.join(SCRIPT_DIR, '..', 'generation', 'year_split_results_2025')
FULL_CSV    = os.path.join(SCRIPT_DIR, 'candidates_WAEMCMC_full.csv')
OUT_PATH    = os.path.join(SCRIPT_DIR, 'cr_prediction_results.xlsx')

CATEGORICAL_COLUMNS = [
    'Test method', 'Deformation Process', 'Heat Treatment Process',
    'Heat Treatment Time', 'Heat Treatment Temperature', 'Immersion Time'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    print("=" * 60)
    print("cr_prediction.py — 初始模型预测腐蚀速率")
    print("=" * 60)

    print("\n[1] 加载预训练模型...")
    with open(os.path.join(MODEL_DIR, 'scaler_X_cutoff2025.pkl'),        'rb') as f: sX       = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'scaler_y_cutoff2025.pkl'),        'rb') as f: sy       = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'label_encoders_cutoff2025.pkl'),  'rb') as f: les      = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'categorical_info_cutoff2025.pkl'),'rb') as f: cat_info = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'num_cols_cutoff2025.pkl'),         'rb') as f: num_cols = pickle.load(f)

    model = CombinedNet(len(num_cols), cat_info, hidden_size=112, dropout_rate=0.12).to(device)
    model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, 'model_cutoff2025.pth'),
        map_location=device, weights_only=True))
    model.eval()
    print(f"    模型加载完成")

    print(f"\n[2] 加载 {FULL_CSV}...")
    df = pd.read_csv(FULL_CSV)
    print(f"    共 {len(df)} 条")

    print("\n[3] 编码特征并预测...")
    print("num_cols:", num_cols)
    print("full csv有这些列吗:", all(c in df.columns for c in num_cols))
    print("缺失的列:", [c for c in num_cols if c not in df.columns])
    X_num = sX.transform(df[num_cols].values.astype(float))
    X_cat = np.column_stack([les[c].transform(df[c].astype(str))
                              for c in CATEGORICAL_COLUMNS])
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_num).to(device),
                     torch.LongTensor(X_cat).to(device)).cpu().numpy()
    print("pred shape:", pred.shape)
    print("pred min/max/mean:", pred.min(), pred.max(), pred.mean())
    pred_cr = sy.inverse_transform(pred).flatten()
    print("pred_cr min/max/mean:", pred_cr.min(), pred_cr.max(), pred_cr.mean())
    pred_cr = np.maximum(pred_cr, 0.0)
    print("after clip min/max:", pred_cr.min(), pred_cr.max())

    result_df = df.copy()
    result_df['Predicted CR (mm/y)'] = pred_cr
    result_df = result_df.sort_values('Predicted CR (mm/y)').reset_index(drop=True)
    result_df.insert(0, 'Rank', range(1, len(result_df) + 1))

    print(f"\n[4] 保存结果: {OUT_PATH}")
    result_df.to_excel(OUT_PATH, index=False)

    wb = load_workbook(OUT_PATH)
    ws = wb.active
    source_col = next((cell.column for cell in ws[1] if cell.value == 'source'), None)
    if source_col:
        for row in ws.iter_rows(min_row=2):
            is_iter = 'iteration' in str(row[source_col - 1].value or '')
            for cell in row:
                cell.font = Font(bold=is_iter)
    ws.freeze_panes = 'A2'
    ws.column_dimensions['A'].width = 6
    ws.column_dimensions['B'].width = 28
    ws.column_dimensions['C'].width = 14
    ws.column_dimensions['D'].width = 20
    wb.save(OUT_PATH)

    n_iter = result_df['source'].str.contains('iteration').sum() if 'source' in result_df.columns else 0
    print(f"    完成！共 {len(result_df)} 条，iteration实验成分 {n_iter} 条（已加粗）")
    print(f"\n预测CR最低的前10名：")
    cols = [c for c in ['Rank','alloy_name','source','Predicted CR (mm/y)'] if c in result_df.columns]
    print(result_df[cols].head(10).to_string(index=False))


if __name__ == '__main__':
    main()