"""
text_feature_removal.py
────────────────────────────────────────────────────────────────────────────────
Remove each text feature column and show predicted corrosion rate changes.

实验逻辑：
  固定模型权重，对 test set 的每个 text feature 列依次做 ablation：
  将该列的 sentence embedding 替换为零向量（相当于去掉该列信息），
  重新前向传播，记录预测腐蚀速率的变化。

  同时做"remove all text features"作为对照，
  与 numerical-only baseline 的比较。

输出：
  1. text_removal_bar_seed{SEED}.png/svg
     — 条形图：去掉每列后预测腐蚀速率的平均绝对变化量
  2. text_removal_values_seed{SEED}.csv
     — 每个样本在各ablation条件下的预测值
  3. 终端打印汇总统计

用法：
  python text_feature_removal.py [SEED]
  默认 SEED=8192
"""

import sys, re, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

# ── Config ─────────────────────────────────────────────────────────────────────
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8192

BASE_DIR   = "D:/vscode/corrosion rate/revision/Interpretability/Transformer/multitext_result_fold3_trimtest_v2/v7_elem_tokens_dm96_nh4_nl2_dff256_drop0p3_wd0p05_ep500_multiseed/fold_3"
DATA_PATH  = "D:/vscode/corrosion rate/revision/Interpretability/Transformer/database_colab_text_pw.xlsx"
MODEL_PATH = f"{BASE_DIR}/seed_{SEED}/best_model.pth"
OUT_DIR    = f"{BASE_DIR}/seed_{SEED}/attention_maps"
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Seed: {SEED}  |  Device: {device}")

# ── Feature columns ────────────────────────────────────────────────────────────
ELEMENT_COLS = ["Mg","Al","Zn","Li","Y","Mn","Nd","Ce","Ca","Gd","Zr","Sc","Fe","Sn","In","La"]
PROP_COLS = [
    "Density","Average Melting Point","Average Boiling Point","Average Atomic Radius",
    "Atomic weight mean","VEC mean","Average Ionization Energy","Average Work Function",
    "Average Fermi Energy","Electronegativity local mismatch","Average IE Difference from Mg",
    "Average Surface Energy","Average Adsorption Energy",
    "Immersion Time (h)","Heat Treatment Temperature (C)","Heat Treatment Time (h)",
]
TEXT_COLUMNS = ["Immersion Time","Deformation Process","Heat Treatment Process",
                "Heat Treatment Time","Heat Treatment Temperature"]
TARGET_COL   = "Corrosion Rate (mm/y)"

# ── DOI sets ───────────────────────────────────────────────────────────────────
FOLD1 = {"10.1016/j.corsci.2020.108958","10.1016/j.jma.2026.102035","10.1016/j.jmst.2023.01.058","10.1016/j.corsci.2017.01.022","10.1016/j.jma.2016.12.002","10.1016/j.corsci.2020.108831","10.1016/S1005-0302(12)60186-8","10.1016/j.corsci.2011.10.033","10.1016/j.corsci.2024.112180","10.1016/j.corsci.2023.111021","10.1016/j.matchar.2022.112121","10.1016/j.corsci.2022.110861","10.1016/j.jallcom.2024.174068","10.1007/s11665-026-13543-5","10.1007/s40195-022-01451-1","10.1016/j.jma.2020.03.014","10.1016/j.jma.2018.09.002"}
FOLD2 = {"10.1038/s41529-026-00755-2","10.1016/j.apsusc.2020.147761","10.1002/sia.7075","10.1016/j.matchar.2021.111294","10.1016/j.electacta.2025.146957","10.1016/j.corcom.2022.10.002","10.1016/j.jpcs.2021.109952","10.1016/j.jallcom.2025.180504","10.1016/j.jallcom.2020.154344","10.1016/j.jallcom.2020.156619","10.1016/j.corsci.2020.108487","10.1016/j.scriptamat.2018.11.046","10.1016/j.jallcom.2020.153839","10.1088/2053-1591/ab51ab","10.1016/j.matdes.2022.110980","10.1007/s11665-020-05034-y"}
FOLD3 = {"10.1016/j.jma.2022.08.003","10.1016/j.jmst.2023.08.068","10.1007/s11661-021-06548-3","10.1016/ j.jma.2022.12.005","10.1016/j.mtcomm.2024.111277","10.1016/j.corsci.2020.108998","10.1016/j.corsci.2020.108451","10.1016/j.jallcom.2026.187172","10.1016/j.jmst.2020.05.067","10.1007/s11665-022-07081-z","10.1002/maco.201609141","10.1016/j.jmst.2025.01.020","10.1016/j.apsusc.2016.01.283","10.1016/j.jma.2020.06.004","10.1016/j.jmrt.2025.04.024","10.20964/2017.11.14"}
MOVE_TO_TRAIN = {"10.1016/j.jmrt.2025.04.024","10.1016/j.corsci.2020.108451","10.1016/j.jmst.2023.08.068","10.1016/j.apsusc.2016.01.283","10.1016/j.jma.2022.08.003","10.1016/j.mtcomm.2024.111277","10.1016/j.jallcom.2026.187172"}
TRAIN_DOIS = FOLD1 | FOLD2 | MOVE_TO_TRAIN
TEST_DOIS  = FOLD3 - MOVE_TO_TRAIN

# ── Preprocessors ──────────────────────────────────────────────────────────────
def parse_hours(text):
    if pd.isna(text): return 0.0
    s = str(text).strip().lower()
    if "no " in s or not s: return 0.0
    m = re.search(r"(\d+\.?\d*)\s*days?", s)
    if m: return float(m.group(1)) * 24
    m = re.search(r"(\d+\.?\d*)\s*(?:hours?|h\b)", s)
    if m: return float(m.group(1))
    m = re.search(r"(\d+\.?\d*)\s*(?:minutes?|mins?|min\b)", s)
    if m: return float(m.group(1)) / 60
    return 0.0

def parse_celsius(text):
    if pd.isna(text): return 0.0
    s = str(text).strip()
    if "no " in s.lower() or not s: return 0.0
    m = re.search(r"(\d+\.?\d*)\s*[Kk]\b", s)
    if m: return float(m.group(1)) - 273.15
    m = re.search(r"(\d+\.?\d*)\s*[°]?[Cc]\b", s)
    if m: return float(m.group(1))
    return 0.0

def standardize_text(text):
    if pd.isna(text): return text
    text = str(text).strip().lower()
    text = re.sub(r"\b(\d+\.?\d*)h\b", r"\1 hours", text)
    for old, new in {"casting":"cast","casted":"cast","rolled":"rolling","extruding":"extruded","extrusion":"extruded","forged":"forging","wrought":"wrought"}.items():
        text = text.replace(old, new)
    if re.search(r"\b\w*cast\w*\b", text, re.IGNORECASE):
        text = re.sub(r"\b\w*cast\w*\b", "cast", text, flags=re.IGNORECASE)
    for old, new in {"solid-solution treament":"solid-solution treatment","ph":"hydrogen evolution test","pw":"weight loss test","pi":"electrochemical test","hydrogen evolution":"hydrogen evolution test","weight loss":"weight loss test","electrochemical":"electrochemical test","potentiodynamic":"electrochemical test","eis":"electrochemical test","tafel":"electrochemical test","t4":"solution-treated","t6":"solution-treated+aging","solution treated":"solution-treated","solution treated + aging":"solution-treated+aging","annealed":"annealing","aging treatment":"aging","aged":"aging"}.items():
        text = text.replace(old, new)
    text = re.sub(r"(\d+)\s*degrees?\s*c", r"\1°C", text)
    text = re.sub(r"(\d+)\s*°\s*c",        r"\1°C", text)
    text = re.sub(r"(\d+)\s*deg\s*c",      r"\1°C", text)
    text = re.sub(r"(\d+)\s*mpa",          r"\1 MPa", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Data loading ───────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(DATA_PATH)
for col in TEXT_COLUMNS:
    if col in df.columns: df[col] = df[col].apply(standardize_text)
df["Immersion Time (h)"]             = df["Immersion Time"].apply(parse_hours)
df["Heat Treatment Temperature (C)"] = df["Heat Treatment Temperature"].apply(parse_celsius)
df["Heat Treatment Time (h)"]        = df["Heat Treatment Time"].apply(parse_hours)
df = df.dropna(subset=[TARGET_COL, "DOI"]).reset_index(drop=True)
for col in ELEMENT_COLS + PROP_COLS:
    if col not in df.columns: df[col] = 0.0
    else: df[col] = df[col].fillna(0.0)

train_df = df[df["DOI"].isin(TRAIN_DOIS)].copy().reset_index(drop=True)
test_df  = df[df["DOI"].isin(TEST_DOIS)].copy().reset_index(drop=True)
print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")

get_log  = lambda sdf: np.log10(sdf[TARGET_COL].values.reshape(-1, 1) + 1e-6)
y_scaler = StandardScaler().fit(get_log(train_df))
elem_sc  = StandardScaler().fit(train_df[ELEMENT_COLS].values.astype(np.float32))
prop_sc  = StandardScaler().fit(train_df[PROP_COLS].values.astype(np.float32))

elem_test = torch.tensor(elem_sc.transform(test_df[ELEMENT_COLS].values.astype(np.float32)), dtype=torch.float32)
prop_test = torch.tensor(prop_sc.transform(test_df[PROP_COLS].values.astype(np.float32)), dtype=torch.float32)

# ── Text encoding ──────────────────────────────────────────────────────────────
print("Encoding text...")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
flat    = [f"{c}:{row[c]}" for _, row in test_df.iterrows() for c in TEXT_COLUMNS]
emb     = encoder.encode(flat, batch_size=64, convert_to_tensor=True,
                          show_progress_bar=False).cpu()
text_test_orig = emb.view(len(test_df), len(TEXT_COLUMNS), emb.shape[-1]).float()
EMB_DIM = emb.shape[-1]   # 384

# ── Model ──────────────────────────────────────────────────────────────────────
def _mlp_proj(in_dim, d_model, dropout):
    hidden = max(d_model, in_dim * 2)
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, d_model), nn.LayerNorm(d_model))

class PreNormTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(d_ff, d_model), nn.Dropout(dropout))
    def forward(self, x):
        n = self.norm1(x); a, _ = self.attn(n, n, n); x = x + a
        return x + self.ff(self.norm2(x))

class TransformerModelV6(nn.Module):
    def __init__(self, num_elem=16, prop_dim=16, num_text=5, text_emb_dim=384,
                 d_model=96, nhead=4, num_layers=2, d_ff=256, dropout=0.3):
        super().__init__()
        self.num_tokens = 1 + num_elem + 1 + num_text
        self.elem_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model)) for _ in range(num_elem)])
        self.prop_proj  = _mlp_proj(prop_dim, d_model, dropout)
        self.text_proj  = nn.Sequential(nn.Linear(text_emb_dim, d_model), nn.LayerNorm(d_model))
        self.cls_token  = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed  = nn.Parameter(torch.empty(1, self.num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.layers = nn.ModuleList([PreNormTransformerLayer(d_model, nhead, d_ff, dropout)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_ff // 2), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(d_ff // 2, 1))
    def forward(self, text_emb, elem_vals, prop_vals):
        B  = text_emb.size(0)
        et = torch.stack([p(elem_vals[:, i:i+1]) for i, p in enumerate(self.elem_projs)], dim=1)
        x  = torch.cat([self.cls_token.expand(B,-1,-1), et,
                        self.prop_proj(prop_vals).unsqueeze(1), self.text_proj(text_emb)], dim=1)
        x  = x + self.pos_embed
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x)[:, 0, :])

model = TransformerModelV6().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
print("Model loaded.")

# ── Helper: predict in original mm/yr space ────────────────────────────────────
def predict_orig(text_tensor):
    """text_tensor: [N, 5, 384] → predicted corrosion rates in mm/yr"""
    with torch.no_grad():
        preds_s = model(text_tensor.to(device),
                        elem_test.to(device),
                        prop_test.to(device)).cpu().numpy().flatten()
    return 10 ** y_scaler.inverse_transform(preds_s.reshape(-1,1)).flatten() - 1e-6

# ── Baseline predictions ───────────────────────────────────────────────────────
pred_baseline = predict_orig(text_test_orig)
y_true = test_df[TARGET_COL].values
r2_baseline = r2_score(
    y_scaler.transform(get_log(test_df)).flatten(),
    y_scaler.transform(np.log10(pred_baseline.reshape(-1,1) + 1e-6)).flatten()
)
print(f"\nBaseline: mean pred = {pred_baseline.mean():.3f} mm/yr  |  R² = {r2_baseline:.4f}")

# ── Remove each text feature (replace with zero vector) ───────────────────────
# 零向量相当于"无信息输入"，text_proj会把它映射到一个固定值，
# 与有内容的embedding相比代表去掉了该列的语义信息
results = {}

print("\n=== Feature Removal Results ===")
print(f"{'Condition':<35} {'Mean pred (mm/yr)':>18} {'Mean |Δpred| (mm/yr)':>22} {'R²':>8}")
print("-" * 85)
print(f"{'Baseline (all features)':<35} {pred_baseline.mean():>18.3f} {'—':>22} {r2_baseline:>8.4f}")

for col_idx, col_name in enumerate(TEXT_COLUMNS):
    text_removed = text_test_orig.clone()
    text_removed[:, col_idx, :] = 0.0   # zero out this column

    pred_removed = predict_orig(text_removed)
    delta = pred_removed - pred_baseline          # signed change per sample
    abs_delta = np.abs(delta)

    results[col_name] = {
        "pred":      pred_removed,
        "delta":     delta,
        "mean_delta":     float(delta.mean()),
        "mean_abs_delta": float(abs_delta.mean()),
        "std_abs_delta":  float(abs_delta.std()),
    }
    print(f"{'Remove ' + col_name:<35} {pred_removed.mean():>18.3f} "
          f"{abs_delta.mean():>22.4f} ± {abs_delta.std():.4f}")

# Remove all text features
text_zeros = torch.zeros_like(text_test_orig)
pred_notext = predict_orig(text_zeros)
delta_notext = pred_notext - pred_baseline
results["All text removed"] = {
    "pred":           pred_notext,
    "delta":          delta_notext,
    "mean_delta":     float(delta_notext.mean()),
    "mean_abs_delta": float(np.abs(delta_notext).mean()),
    "std_abs_delta":  float(np.abs(delta_notext).std()),
}
print(f"{'Remove ALL text features':<35} {pred_notext.mean():>18.3f} "
      f"{np.abs(delta_notext).mean():>22.4f} ± {np.abs(delta_notext).std():.4f}")

# ── Save CSV ───────────────────────────────────────────────────────────────────
rows = []
for i in range(len(test_df)):
    row = {"sample_idx": i,
           "doi": test_df["DOI"].iloc[i],
           "true_cr": float(y_true[i]),
           "pred_baseline": float(pred_baseline[i])}
    for col_name, res in results.items():
        key = col_name.replace(" ", "_").lower()
        row[f"pred_{key}"]  = float(res["pred"][i])
        row[f"delta_{key}"] = float(res["delta"][i])
    rows.append(row)

csv_path = os.path.join(OUT_DIR, f"text_removal_values_seed{SEED}.csv")
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"\nSaved → {csv_path}")

# ── Plot: mean absolute change per removed feature ────────────────────────────
labels     = list(TEXT_COLUMNS) + ["All text\nremoved"]
mean_abs   = [results[c]["mean_abs_delta"] for c in TEXT_COLUMNS] + \
             [results["All text removed"]["mean_abs_delta"]]
std_abs    = [results[c]["std_abs_delta"]  for c in TEXT_COLUMNS] + \
             [results["All text removed"]["std_abs_delta"]]

label_display = ["Immersion\nTime", "Deformation\nProcess", "HT Process",
                 "HT Time", "HT Temp", "All text\nremoved"]
colors = ['#4C72B0'] * 5 + ['#C44E52']

fig, ax = plt.subplots(figsize=(6, 5))

bars = ax.bar(range(len(labels)), mean_abs,
              color=colors, width=0.6,
              edgecolor='white', linewidth=0.5)

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(label_display, fontsize=7)
ax.set_ylabel('Mean |Δ predicted corrosion rate|\n(mm yr$^{-1}$)', fontsize=8, labelpad=4)
ax.tick_params(labelsize=7, width=0.8, length=3)
for sp in ax.spines.values():
    sp.set_visible(True); sp.set_linewidth(0.8)
ax.grid(axis='y', linewidth=0.4, alpha=0.4)
ax.set_axisbelow(True)

# 标注数值
for i, (v, s) in enumerate(zip(mean_abs, std_abs)):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

fig.tight_layout()
out_png = os.path.join(OUT_DIR, f"text_removal_bar_seed{SEED}.png")
out_svg = os.path.join(OUT_DIR, f"text_removal_bar_seed{SEED}.svg")
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(out_svg, bbox_inches='tight',          facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Saved → {out_png}")

print("\n✅ Done.")
print(f"   Bar chart : {out_png}")
print(f"   CSV values: {csv_path}")