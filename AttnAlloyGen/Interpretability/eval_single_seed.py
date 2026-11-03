import sys, re, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 8192

BASE_DIR  = "D:/vscode/corrosion rate/revision/Interpretability/Transformer/multitext_result_fold3_trimtest_v2/v7_elem_tokens_dm96_nh4_nl2_dff256_drop0p3_wd0p05_ep500_multiseed/fold_3"
DATA_PATH = "D:/vscode/corrosion rate/revision/Interpretability/Transformer/database_colab_text_pw.xlsx"
MODEL_PATH = f"{BASE_DIR}/seed_{SEED}/best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Seed: {SEED}  |  Model: {MODEL_PATH}  |  Device: {device}")

# ── Feature columns (must match training script) ──────────────────────────────
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
TARGET_COL = "Corrosion Rate (mm/y)"

# ── DOI sets (must match training script) ─────────────────────────────────────
FOLD1 = {"10.1016/j.corsci.2020.108958","10.1016/j.jma.2026.102035","10.1016/j.jmst.2023.01.058","10.1016/j.corsci.2017.01.022","10.1016/j.jma.2016.12.002","10.1016/j.corsci.2020.108831","10.1016/S1005-0302(12)60186-8","10.1016/j.corsci.2011.10.033","10.1016/j.corsci.2024.112180","10.1016/j.corsci.2023.111021","10.1016/j.matchar.2022.112121","10.1016/j.corsci.2022.110861","10.1016/j.jallcom.2024.174068","10.1007/s11665-026-13543-5","10.1007/s40195-022-01451-1","10.1016/j.jma.2020.03.014","10.1016/j.jma.2018.09.002"}
FOLD2 = {"10.1038/s41529-026-00755-2","10.1016/j.apsusc.2020.147761","10.1002/sia.7075","10.1016/j.matchar.2021.111294","10.1016/j.electacta.2025.146957","10.1016/j.corcom.2022.10.002","10.1016/j.jpcs.2021.109952","10.1016/j.jallcom.2025.180504","10.1016/j.jallcom.2020.154344","10.1016/j.jallcom.2020.156619","10.1016/j.corsci.2020.108487","10.1016/j.scriptamat.2018.11.046","10.1016/j.jallcom.2020.153839","10.1088/2053-1591/ab51ab","10.1016/j.matdes.2022.110980","10.1007/s11665-020-05034-y"}
FOLD3 = {"10.1016/j.jma.2022.08.003","10.1016/j.jmst.2023.08.068","10.1007/s11661-021-06548-3","10.1016/ j.jma.2022.12.005","10.1016/j.mtcomm.2024.111277","10.1016/j.corsci.2020.108998","10.1016/j.corsci.2020.108451","10.1016/j.jallcom.2026.187172","10.1016/j.jmst.2020.05.067","10.1007/s11665-022-07081-z","10.1002/maco.201609141","10.1016/j.jmst.2025.01.020","10.1016/j.apsusc.2016.01.283","10.1016/j.jma.2020.06.004","10.1016/j.jmrt.2025.04.024","10.20964/2017.11.14"}
MOVE_TO_TRAIN = {"10.1016/j.jmrt.2025.04.024","10.1016/j.corsci.2020.108451","10.1016/j.jmst.2023.08.068","10.1016/j.apsusc.2016.01.283","10.1016/j.jma.2022.08.003","10.1016/j.mtcomm.2024.111277","10.1016/j.jallcom.2026.187172"}

TRAIN_DOIS = FOLD1 | FOLD2 | MOVE_TO_TRAIN
TEST_DOIS  = FOLD3 - MOVE_TO_TRAIN

# ── Preprocessors ─────────────────────────────────────────────────────────────
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
    text = re.sub(r"(\d+)\s*°\s*c", r"\1°C", text)
    text = re.sub(r"(\d+)\s*deg\s*c", r"\1°C", text)
    text = re.sub(r"(\d+)\s*mpa", r"\1 MPa", text)
    return re.sub(r"\s+", " ", text).strip()

# ── Load & preprocess data ────────────────────────────────────────────────────
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
print(f"Train: {len(train_df)} samples ({len(TRAIN_DOIS)} DOIs)  |  Test: {len(test_df)} samples ({len(TEST_DOIS)} DOIs)")

# ── Fit scalers on train (must match training) ────────────────────────────────
get_log = lambda sdf: np.log10(sdf[TARGET_COL].values.reshape(-1, 1) + 1e-6)
y_scaler   = StandardScaler().fit(get_log(train_df))
elem_sc    = StandardScaler().fit(train_df[ELEMENT_COLS].values.astype(np.float32))
prop_sc    = StandardScaler().fit(train_df[PROP_COLS].values.astype(np.float32))

y_test    = torch.tensor(y_scaler.transform(get_log(test_df)), dtype=torch.float32)
elem_test = torch.tensor(elem_sc.transform(test_df[ELEMENT_COLS].values.astype(np.float32)), dtype=torch.float32)
prop_test = torch.tensor(prop_sc.transform(test_df[PROP_COLS].values.astype(np.float32)), dtype=torch.float32)

# ── Text encoding ─────────────────────────────────────────────────────────────
print("Encoding text...")
encoder = SentenceTransformer("all-MiniLM-L6-v2", device=str(device))
flat = [f"{c}:{row[c]}" for _, row in test_df.iterrows() for c in TEXT_COLUMNS]
emb = encoder.encode(flat, batch_size=64, convert_to_tensor=True, show_progress_bar=False).cpu()
text_test = emb.view(len(test_df), len(TEXT_COLUMNS), emb.shape[-1]).float()

# ── Dataset / DataLoader ──────────────────────────────────────────────────────
class CorrosionDataset(Dataset):
    def __init__(self, te, ev, pv, lb):
        self.te=te; self.ev=ev; self.pv=pv; self.lb=lb
    def __len__(self): return len(self.lb)
    def __getitem__(self, i): return self.te[i], self.ev[i], self.pv[i], self.lb[i]

loader = DataLoader(CorrosionDataset(text_test, elem_test, prop_test, y_test),
                    batch_size=16, shuffle=False)

# ── Model (identical to training script) ─────────────────────────────────────
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
        B = text_emb.size(0)
        et = torch.stack([p(elem_vals[:, i:i+1]) for i, p in enumerate(self.elem_projs)], dim=1)
        x  = torch.cat([self.cls_token.expand(B,-1,-1), et,
                        self.prop_proj(prop_vals).unsqueeze(1), self.text_proj(text_emb)], dim=1)
        x  = x + self.pos_embed
        for layer in self.layers: x = layer(x)
        return self.head(self.norm(x)[:, 0, :])

# ── Load model & predict ──────────────────────────────────────────────────────
model = TransformerModelV6().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

preds, tgts = [], []
with torch.no_grad():
    for te, ev, pv, lb in loader:
        te, ev, pv = te.to(device), ev.to(device), pv.to(device)
        preds.extend(model(te, ev, pv).cpu().numpy())
        tgts.extend(lb.numpy())

preds = np.array(preds).reshape(-1)
tgts  = np.array(tgts).reshape(-1)

r2  = r2_score(tgts, preds)
mse = mean_squared_error(tgts, preds)

print(f"\n{'='*40}")
print(f"Seed {SEED}  →  R² = {r2:.4f}  |  MSE = {mse:.4f}")
print(f"Test samples: {len(tgts)}  |  Test DOIs: {len(TEST_DOIS)}")
print(f"{'='*40}")

# Per-DOI breakdown
test_df = test_df.copy()
test_df["pred"] = preds
test_df["tgt"]  = tgts
test_df["err"]  = np.abs(preds - tgts)
doi_stats = test_df.groupby("DOI").agg(n=("err","count"), mean_err=("err","mean")).sort_values("mean_err", ascending=False)
print("\nPer-DOI mean absolute error:")
print(doi_stats.to_string())

# ── Scatter plot (NC style, matching year_split_replay) ───────────────────────
# Inverse-transform predictions and targets back to original mm/yr space
preds_orig = 10 ** y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten() - 1e-6
tgts_orig  = 10 ** y_scaler.inverse_transform(tgts.reshape(-1, 1)).flatten()  - 1e-6

from sklearn.metrics import mean_squared_error as _mse
rmse_orig = float(np.sqrt(_mse(tgts_orig, preds_orig)))
pd.DataFrame({'DOI': test_df['DOI'].values,
              'experimental_cr': tgts_orig,
              'predicted_cr': preds_orig}).to_csv(
    os.path.join(os.path.dirname(MODEL_PATH), f'parity_seed{SEED}_data.csv'), index=False)
print('R2 in original space:', r2_score(tgts_orig, preds_orig))
# R² in log-transformed space (consistent with training metric)
# RMSE in original mm/yr space (physically interpretable)
r2_orig   = r2

all_vals = np.concatenate([tgts_orig, preds_orig])
vmax = float(np.ceil(all_vals.max() * 1.08))
vmax = max(vmax, 5.0)

fig_w = 7.5 / 2.54
fig_h = 7.5 / 2.54
fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

# Test points — single color (no year coloring here)
ax.scatter(
    preds_orig, tgts_orig,
    color='#4C72B0', s=35, alpha=0.90,
    edgecolors='none', zorder=3,
    label='Test',
)

# Parity line
ax.plot([0, vmax], [0, vmax],
        color='#555', linewidth=1.2,
        linestyle=(0, (4, 3)), zorder=2)

ax.set_xlim(0, vmax)
ax.set_ylim(0, vmax)
ax.set_aspect('equal', adjustable='box')
ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, integer=True))
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.tick_params(labelsize=8, width=1.0, length=3.5)

for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(1.0)

ax.grid(False)
ax.set_xlabel('Predicted corrosion rate (mm\u2009yr$^{-1}$)', fontsize=9, labelpad=4)
ax.set_ylabel('Experimental corrosion rate (mm\u2009yr$^{-1}$)', fontsize=9, labelpad=4)

stat_lines = [
    f'Test $R^2$ = {r2_orig:.3f}',
    f'RMSE = {rmse_orig:.3f} mm\u2009yr$^{{-1}}$',
]
ax.text(1.04, 0.5, '\n'.join(stat_lines),
        transform=ax.transAxes,
        fontsize=8, va='center', ha='left',
        linespacing=1.8)

ax.legend(fontsize=7.5, loc='upper left', frameon=False,
          labelspacing=0.35, handletextpad=0.4, markerscale=1.1)

out_dir = os.path.dirname(MODEL_PATH)
out_png = os.path.join(out_dir, f'parity_seed{SEED}.png')
out_svg = os.path.join(out_dir, f'parity_seed{SEED}.svg')
fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
fig.savefig(out_svg, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close(fig)
print(f"\nSaved → {out_png}")
print(f"Saved → {out_svg}")