"""
attention_viz.py
────────────────────────────────────────────────────────────────────────────────
针对 TransformerModelV6 的 attention 可视化脚本。

Token 顺序（与 training script 完全一致）：
  pos 0        : [CLS]
  pos 1 ~ 16   : element tokens  (Mg, Al, Zn, Li, Y, Mn, Nd, Ce, Ca, Gd, Zr, Sc, Fe, Sn, In, La)
  pos 17       : prop token
  pos 18 ~ 22  : text tokens     (Immersion Time, Deformation Process,
                                  Heat Treatment Process, Heat Treatment Time,
                                  Heat Treatment Temperature)

可视化内容：
  - 最后一层 Transformer 的 attention weights（4 个 head 平均）
  - 提取 element tokens → text tokens 的子矩阵 (16 × 5)
  - 对 test set 所有样本取平均，绘制平均 attention heatmap
  - 同时保存每个 head 的 average heatmap（4 张子图）

用法：
  python attention_viz.py [SEED]
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
TEXT_LABELS  = ["Immersion\nTime", "Deformation\nProcess", "Heat Treatment\nProcess",
                "Heat Treatment\nTime", "Heat Treatment\nTemperature"]
TARGET_COL   = "Corrosion Rate (mm/y)"

# Token position constants
POS_CLS        = 0
POS_ELEM_START = 1                          # 1 ~ 16
POS_ELEM_END   = 1 + len(ELEMENT_COLS)      # 17 (exclusive)
POS_PROP       = POS_ELEM_END               # 17
POS_TEXT_START = POS_ELEM_END + 1           # 18
POS_TEXT_END   = POS_TEXT_START + len(TEXT_COLUMNS)  # 23 (exclusive)

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

y_test    = torch.tensor(y_scaler.transform(get_log(test_df)), dtype=torch.float32)
elem_test = torch.tensor(elem_sc.transform(test_df[ELEMENT_COLS].values.astype(np.float32)), dtype=torch.float32)
prop_test = torch.tensor(prop_sc.transform(test_df[PROP_COLS].values.astype(np.float32)), dtype=torch.float32)

print("Encoding text...")
encoder  = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
flat     = [f"{c}:{row[c]}" for _, row in test_df.iterrows() for c in TEXT_COLUMNS]
emb      = encoder.encode(flat, batch_size=64, convert_to_tensor=True, show_progress_bar=False).cpu()
text_test = emb.view(len(test_df), len(TEXT_COLUMNS), emb.shape[-1]).float()

class CorrosionDataset(Dataset):
    def __init__(self, te, ev, pv, lb):
        self.te=te; self.ev=ev; self.pv=pv; self.lb=lb
    def __len__(self): return len(self.lb)
    def __getitem__(self, i): return self.te[i], self.ev[i], self.pv[i], self.lb[i]

# batch_size=1 以便逐样本收集 attention
loader = DataLoader(CorrosionDataset(text_test, elem_test, prop_test, y_test),
                    batch_size=1, shuffle=False)

# ── Model definition ───────────────────────────────────────────────────────────
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
        n = self.norm1(x)
        # need_weights=True, average_attn_weights=False → shape [B, nhead, seq, seq]
        a, attn_w = self.attn(n, n, n, need_weights=True, average_attn_weights=False)
        x = x + a
        return x + self.ff(self.norm2(x)), attn_w   # 返回 attn weights

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
        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x)       # 只保留最后一层的 attn
        return self.head(self.norm(x)[:, 0, :]), last_attn

# ── Load model ─────────────────────────────────────────────────────────────────
model = TransformerModelV6().to(device)

# 加载权重时需要忽略 forward 签名变化（权重结构没变）
state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state)
model.eval()
print("Model loaded.")

# ── Collect attention weights ──────────────────────────────────────────────────
# attn shape per sample: [1, nhead=4, seq_len=23, seq_len=23]
# 提取 element → text 子矩阵: rows [1:17], cols [18:23]  → shape [4, 16, 5]

print("Collecting attention weights...")
all_attn = []   # list of np arrays, shape [4, 16, 5]

with torch.no_grad():
    for te, ev, pv, _ in loader:
        te, ev, pv = te.to(device), ev.to(device), pv.to(device)
        _, attn_w = model(te, ev, pv)
        # attn_w: [1, nhead, seq, seq]
        attn_np = attn_w[0].cpu().numpy()   # [nhead, seq, seq]
        sub = attn_np[:, POS_ELEM_START:POS_ELEM_END, POS_TEXT_START:POS_TEXT_END]
        # sub: [nhead, 16, 5]
        all_attn.append(sub)

all_attn = np.stack(all_attn, axis=0)   # [N, nhead, 16, 5]
print(f"Collected: {all_attn.shape}  →  [samples, heads, elements, text_features]")

# ── Average attention maps ─────────────────────────────────────────────────────
avg_per_head = all_attn.mean(axis=0)    # [nhead, 16, 5]
avg_all      = avg_per_head.mean(axis=0) # [16, 5]  — head-averaged & sample-averaged

# ── Plot 1: Head-averaged mean heatmap (主图，用于论文) ─────────────────────────
def _format_heatmap(ax, data, xlabels, ylabels, title, vmin=None, vmax=None, cmap='YlOrRd'):
    """绘制单张 heatmap，返回 im 供 colorbar 用"""
    im = ax.imshow(data, cmap=cmap, aspect='auto',
                   vmin=vmin if vmin is not None else data.min(),
                   vmax=vmax if vmax is not None else data.max(),
                   interpolation='nearest')
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=7, ha='center')
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    # 网格线
    ax.set_xticks(np.arange(-0.5, len(xlabels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ylabels), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.6)
    ax.tick_params(which='minor', length=0)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.8)
    return im

# 转置：Process features → y轴，Elements → x轴，元素顺序反转（La→Mg，与图2一致）
# avg_all shape: [16 elem, 5 text]  →  转置后 [5 text, 16 elem]，再左右翻转元素列
avg_all_T         = avg_all.T[:, ::-1]      # [5, 16]，元素从 La 到 Mg
elem_labels_rev   = ELEMENT_COLS[::-1]      # ['La','In','Sn',...,'Mg']
text_labels_plain = ["Immersion Time", "Deformation Process",
                     "Heat Treatment Process", "Heat Treatment Time",
                     "Heat Treatment Temperature"]

# 图幅：横向宽图，仿图2比例
fig_w = 17.8 / 2.54   # 双栏宽
fig_h =  5.0 / 2.54
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

im = _format_heatmap(ax, avg_all_T,
                     xlabels=elem_labels_rev,
                     ylabels=text_labels_plain,
                     title='')

# x轴标签放顶部（仿图2）
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()
ax.set_xlabel('Elements', fontsize=8, labelpad=4, fontweight='bold')
ax.set_ylabel('Process Features', fontsize=8, labelpad=4, fontweight='bold')

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.ax.tick_params(labelsize=7)
cbar.set_label('Attention Value', fontsize=7, labelpad=4)

fig.tight_layout()
out_avg_png = os.path.join(OUT_DIR, f'attn_avg_seed{SEED}.png')
out_avg_svg = os.path.join(OUT_DIR, f'attn_avg_seed{SEED}.svg')
fig.savefig(out_avg_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(out_avg_svg, bbox_inches='tight',          facecolor='white', edgecolor='none')
plt.close(fig)
print(f"Saved → {out_avg_png}")

# ── Plot 2: 4 heads × individual heatmaps (辅助图，供 SI 或审稿) ───────────────
# 同样转置：[nhead, 16, 5] → 每个head取 [5, 16] 并翻转元素列
nhead = avg_per_head.shape[0]
avg_per_head_T = avg_per_head.transpose(0, 2, 1)[:, :, ::-1]  # [nhead, 5, 16]

fig2, axes = plt.subplots(nhead, 1, figsize=(17.8 / 2.54, nhead * 3.5 / 2.54),
                           sharex=True)

vmin_h = avg_per_head_T.min()
vmax_h = avg_per_head_T.max()

for h, ax_h in enumerate(axes):
    im_h = _format_heatmap(ax_h, avg_per_head_T[h],
                            xlabels=elem_labels_rev if h == 0 else [],
                            ylabels=text_labels_plain,
                            title=f'Head {h+1}',
                            vmin=vmin_h, vmax=vmax_h)
    ax_h.set_ylabel('Process\nFeatures', fontsize=7, labelpad=4)
    if h == 0:
        ax_h.xaxis.set_label_position('top')
        ax_h.xaxis.tick_top()
        ax_h.set_xlabel('Elements', fontsize=7, labelpad=3, fontweight='bold')

fig2.colorbar(im_h, ax=axes[-1], fraction=0.02, pad=0.02).ax.tick_params(labelsize=6)
fig2.tight_layout()
out_heads_png = os.path.join(OUT_DIR, f'attn_per_head_seed{SEED}.png')
out_heads_svg = os.path.join(OUT_DIR, f'attn_per_head_seed{SEED}.svg')
fig2.savefig(out_heads_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig2.savefig(out_heads_svg, bbox_inches='tight',           facecolor='white', edgecolor='none')
plt.close(fig2)
print(f"Saved → {out_heads_png}")

# ── Save raw attention values to CSV ──────────────────────────────────────────
rows = []
for elem_i, elem in enumerate(ELEMENT_COLS):
    for text_j, tcol in enumerate(TEXT_COLUMNS):
        rows.append({
            "element":       elem,
            "text_feature":  tcol,
            "attn_mean":     float(avg_all[elem_i, text_j]),
            **{f"head{h+1}_mean": float(avg_per_head[h, elem_i, text_j])
               for h in range(nhead)},
        })
csv_path = os.path.join(OUT_DIR, f'attn_values_seed{SEED}.csv')
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"Saved → {csv_path}")

print("\n✅ Attention visualization complete.")
print(f"   Main heatmap  : {out_avg_png}")
print(f"   Per-head maps : {out_heads_png}")
print(f"   CSV values    : {csv_path}")