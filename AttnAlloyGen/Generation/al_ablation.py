import os, re, sys, random, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.manifold import TSNE
from scipy.stats import norm as scipy_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import WenAlloys

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(SCRIPT_DIR, 'database_colab_text_new1.xlsx')
WAE_WEIGHTS = os.path.join(SCRIPT_DIR, 'WAE_model_weights_pw.pth')

# ── year_split 预训练模型路径 ──────────────────────────
YEAR_SPLIT_DIR     = os.path.join(SCRIPT_DIR, 'year_split_results_2025')
PRETRAINED_MODEL   = os.path.join(YEAR_SPLIT_DIR, 'model_cutoff2025.pth')
PRETRAINED_SX      = os.path.join(YEAR_SPLIT_DIR, 'scaler_X_cutoff2025.pkl')
PRETRAINED_SY      = os.path.join(YEAR_SPLIT_DIR, 'scaler_y_cutoff2025.pkl')
PRETRAINED_LES     = os.path.join(YEAR_SPLIT_DIR, 'label_encoders_cutoff2025.pkl')
PRETRAINED_CATINFO = os.path.join(YEAR_SPLIT_DIR, 'categorical_info_cutoff2025.pkl')
PRETRAINED_NUMCOLS = os.path.join(YEAR_SPLIT_DIR, 'num_cols_cutoff2025.pkl')

# ── 最优预测模型参数（来自year_split cutoff=2025结果） ─────────
BEST_PARAMS = {'hidden_size': 112, 'dropout_rate': 0.12,
               'learning_rate': 0.0006, 'seed': 119700}

# ── 常量 ──────────────────────────────────────────────────────
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
ELEMENTS   = ['Mg','Al','Zn','Li','Y','Mn','Nd','Ce','Ca','Gd','Zr','Sc','Fe','Sn','In','La']
CATEGORICAL_COLUMNS = [
    'Test method','Deformation Process','Heat Treatment Process',
    'Heat Treatment Time','Heat Treatment Temperature','Immersion Time'
]

EXCLUDED_DOIS = {
    '10.1007/s11665-025-13079-0','10.1007/s11665-026-13708-2',
    '10.1016/j.jma.2018.11.004', '10.1016/j.jmrt.2022.05.161',
}
DOI_YEAR_MAP = {
    '10.1002/maco.201609141':2016,'10.1002/sia.7075':2022,
    '10.1007/s11661-021-06548-3':2021,'10.1007/s11665-020-05034-y':2020,
    '10.1007/s11665-022-07081-z':2022,'10.1007/s11665-026-13543-5':2026,
    '10.1007/s40195-022-01451-1':2022,'10.1016/ j.jma.2022.12.005':2022,
    '10.1016/S1005-0302(12)60186-8':2012,'10.1016/j.apsusc.2016.01.283':2016,
    '10.1016/j.apsusc.2020.147761':2020,'10.1016/j.corcom.2022.10.002':2022,
    '10.1016/j.corsci.2011.10.033':2011,'10.1016/j.corsci.2017.01.022':2017,
    '10.1016/j.corsci.2020.108451':2020,'10.1016/j.corsci.2020.108487':2020,
    '10.1016/j.corsci.2020.108831':2020,'10.1016/j.corsci.2020.108958':2020,
    '10.1016/j.corsci.2020.108998':2020,'10.1016/j.corsci.2022.110861':2022,
    '10.1016/j.corsci.2023.111021':2023,'10.1016/j.corsci.2024.112180':2024,
    '10.1016/j.electacta.2025.146957':2025,'10.1016/j.jallcom.2020.153839':2020,
    '10.1016/j.jallcom.2020.154344':2020,'10.1016/j.jallcom.2020.156619':2020,
    '10.1016/j.jallcom.2024.174068':2024,'10.1016/j.jallcom.2025.180504':2025,
    '10.1016/j.jallcom.2026.187172':2026,'10.1016/j.jma.2016.12.002':2016,
    '10.1016/j.jma.2018.09.002':2018,'10.1016/j.jma.2020.03.014':2020,
    '10.1016/j.jma.2020.06.004':2020,'10.1016/j.jma.2022.08.003':2022,
    '10.1016/j.jma.2026.102035':2026,'10.1016/j.jmrt.2025.04.024':2025,
    '10.1016/j.jmst.2020.05.067':2020,'10.1016/j.jmst.2023.01.058':2023,
    '10.1016/j.jmst.2023.08.068':2023,'10.1016/j.jmst.2025.01.020':2025,
    '10.1016/j.jpcs.2021.109952':2021,'10.1016/j.matchar.2021.111294':2021,
    '10.1016/j.matchar.2022.112121':2022,'10.1016/j.matdes.2022.110980':2022,
    '10.1016/j.mtcomm.2024.111277':2024,'10.1016/j.scriptamat.2018.11.046':2018,
    '10.1038/s41529-026-00755-2':2026,'10.1088/2053-1591/ab51ab':2019,
    '10.11900/0412.1961.2024.00230':2024,'10.20964/2017.11.14':2017,
    '10.1016/j.jallcom.2020.158493':2020,'10.1016/S1003-6326(18)64696-5':2018,
}

# ── 主动学习参数 ──────────────────────────────────────────────
N_INIT       = 30    # 初始训练集大小
BATCH_SIZE   = 2     # 每轮选点数（对应论文每轮2-4个实验）
N_ITERATIONS = 10    # 迭代轮数
N_TRIALS     = 5     # 重复次数（取均值±std）
N_CANDIDATES = 300   # 每轮WAE+MCMC生成候选数
MC_SAMPLES   = 30    # MC Dropout前传次数
UCB_KAPPA    = 1.0   # UCB探索系数（标准值）
TRAIN_EPOCHS = 600
TRAIN_PATIENCE = 150

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 1. 物理描述符计算（移植自 generation_quality.py）
# ══════════════════════════════════════════════════════════════
_WF   = {'Sr':2.41,'Eu':2.43,'Ca':2.6,'La':2.64,'Sm':2.74,'Nd':2.86,'Pr':2.88,'Ho':2.91,
         'Li':2.92,'Er':2.92,'Tm':2.95,'Gd':2.96,'Ce':2.98,'Tb':2.98,'Dy':2.98,'Y':2.99,
         'Lu':3.03,'Sc':3.12,'Ti':3.5,'Hf':3.54,'Mg':3.55,'Zr':3.57,'In':3.7,'Ga':3.98,
         'Zn':4.01,'Al':4.04,'Bi':4.06,'Sn':4.07,'V':4.11,'Cr':4.11,'Mo':4.11,'Nb':4.15,
         'Ta':4.26,'Sb':4.3,'Fe':4.31,'Mn':4.32,'Ge':4.39,'As':4.4,'W':4.47,'Te':4.51,
         'Co':4.52,'Cu':4.53,'B':4.67,'Si':4.73,'Be':4.87,'Ni':4.98}
_MP   = {'Sr':777,'Eu':822,'Ca':842,'La':920,'Sm':1072,'Nd':1021,'Pr':931,'Ho':1470,
         'Li':180.5,'Er':1529,'Tm':1545,'Gd':1312,'Ce':795,'Tb':1356,'Dy':1407,'Y':1526,
         'Lu':1652,'Sc':1541,'Ti':1668,'Hf':2233,'Mg':650,'Zr':1855,'In':156.6,'Ga':29.8,
         'Zn':419.5,'Al':660.3,'Bi':271.4,'Sn':231.9,'V':1910,'Cr':1907,'Mo':2623,'Nb':2477,
         'Ta':3017,'Sb':630.6,'Fe':1538,'Mn':1246,'Ge':938.3,'As':817,'W':3422,'Te':449.5,
         'Co':1495,'Cu':1084.6,'B':2076,'Si':1414,'Be':1287,'Ni':1455}
_BP   = {'Sr':1382,'Eu':1529,'Ca':1484,'La':3464,'Sm':1794,'Nd':3074,'Pr':3290,'Ho':2700,
         'Li':1342,'Er':2868,'Tm':1950,'Gd':3273,'Ce':3426,'Tb':3230,'Dy':2562,'Y':3338,
         'Lu':3402,'Sc':2830,'Ti':3287,'Hf':4602,'Mg':1091,'Zr':4409,'In':2072,'Ga':2204,
         'Zn':907,'Al':2470,'Bi':1564,'Sn':2602,'V':3407,'Cr':2671,'Mo':4639,'Nb':4744,
         'Ta':5455,'Sb':1587,'Fe':2862,'Mn':2061,'Ge':2820,'As':614,'W':5555,'Te':988,
         'Co':2927,'Cu':2562,'B':4000,'Si':3265,'Be':2471,'Ni':2913}
_FE   = {'Sr':1.15,'Eu':0.95,'Ca':1.34,'La':1.65,'Sm':0.92,'Nd':1.21,'Pr':1.05,'Ho':1.50,
         'Li':4.74,'Er':1.21,'Tm':1.30,'Gd':1.10,'Ce':1.11,'Tb':1.12,'Dy':1.20,'Y':1.58,
         'Lu':1.20,'Sc':1.56,'Ti':1.63,'Hf':1.45,'Mg':7.08,'Zr':1.58,'In':1.34,'Ga':1.63,
         'Zn':7.50,'Al':11.70,'Bi':1.35,'Sn':1.13,'V':1.30,'Cr':1.48,'Mo':0.68,'Nb':1.40,
         'Ta':1.53,'Sb':1.15,'Fe':2.20,'Mn':1.19,'Ge':1.60,'As':1.11,'W':1.34,'Te':0.92,
         'Co':1.49,'Cu':7.00,'B':2.90,'Si':1.25,'Be':14.30,'Ni':5.10}
_DENS = {'Sr':2.64,'Eu':5.244,'Ca':2.6,'La':6.145,'Sm':7.52,'Nd':7.01,'Pr':6.773,'Ho':8.795,
         'Li':0.534,'Er':9.066,'Tm':9.321,'Gd':7.895,'Ce':6.77,'Tb':8.229,'Dy':8.55,'Y':4.469,
         'Lu':9.84,'Sc':2.989,'Ti':4.506,'Hf':13.31,'Mg':1.738,'Zr':6.52,'In':7.31,'Ga':5.904,
         'Zn':7.14,'Al':2.70,'Sn':7.265,'Fe':7.87,'Mn':7.21,'Co':8.86,'Cu':8.96,'Ni':8.912}
_SE   = {'Sr':0.35,'Eu':0.49,'Ca':0.47,'La':0.73,'Sm':0.88,'Nd':0.87,'Pr':0.83,'Ho':1.01,
         'Li':0.5,'Er':1.04,'Gd':0.86,'Ce':1.09,'Tb':0.98,'Dy':1.01,'Y':1.0,'Lu':1.11,
         'Sc':1.25,'Ti':2.0,'Mg':0.59,'Zr':1.61,'In':0.31,'Zn':0.44,'Al':0.8,'Sn':0.55,
         'Nb':0.87,'Fe':2.53,'Mn':3.49,'Co':2.34,'Bi':0.24}
_ADS  = {'Sr':-0.337,'Eu':-0.520,'Ca':-0.375,'La':-0.612,'Sm':-0.744,'Nd':4.336,
         'Pr':-0.778,'Ho':-0.696,'Li':-0.286,'Er':-1.025,'Gd':-0.639,'Ce':-0.596,
         'Tb':-0.680,'Dy':-0.604,'Y':-0.757,'Lu':-1.151,'Sc':-0.783,'Ti':-0.784,
         'Mg':0.106,'Zr':-0.743,'In':0.569,'Zn':0.253,'Al':0.196,'Sn':-0.858,
         'Nb':-0.644,'Fe':-0.883,'Mn':-0.826,'Co':-0.578,'Bi':0.715}

# 预计算固定数组
_ie_arr  = np.array([Element(e).ionization_energies[0] for e in ELEMENTS])
_wf_arr  = np.array([_WF.get(e, 0) for e in ELEMENTS])
_mp_arr  = np.array([_MP.get(e, 0) for e in ELEMENTS])
_bp_arr  = np.array([_BP.get(e, 0) for e in ELEMENTS])
_fe_arr  = np.array([_FE.get(e, 0) for e in ELEMENTS])
_se_arr  = np.array([_SE.get(e, 0) for e in ELEMENTS])
_ads_arr = np.array([_ADS.get(e, 0) for e in ELEMENTS])
_ar_arr  = np.array([Element(e).atomic_radius.real for e in ELEMENTS])
_mg_ie   = Element('Mg').ionization_energies[0]
_dens_aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
_dens_d  = np.array([_DENS.get(e, 1) for e in ELEMENTS])
_wa      = WenAlloys()

DESCRIPTOR_COLS = [
    'Atomic weight mean', 'Electronegativity local mismatch', 'VEC mean',
    'Average Ionization Energy', 'Average Work Function',
    'Average IE Difference from Mg', 'Density', 'Average Melting Point',
    'Average Boiling Point', 'Average Fermi Energy', 'Average Atomic Radius',
    'Average Surface Energy', 'Average Adsorption Energy',
]


def compute_descriptors(comp_array):
    """
    输入: comp_array (N, 16) 原子分数（已归一化）
    输出: DataFrame，列名与数据集数值特征完全一致
    """
    records = []
    for comp in comp_array:
        comp = np.clip(comp, 0, None)
        s = comp.sum()
        if s < 1e-10:
            comp = np.zeros(len(ELEMENTS)); comp[0] = 1.0
        else:
            comp = comp / s
        formula_parts = [f"{e}{comp[i]:.6f}"
                         for i, e in enumerate(ELEMENTS) if comp[i] > 1e-8]
        c = Composition(''.join(formula_parts))
        wa_res = dict(zip(_wa.feature_labels(), _wa.featurize(c)))
        molar_mass = np.dot(comp, _dens_aw)
        molar_vol  = sum(comp[i] * _dens_aw[i] / _dens_d[i]
                         for i in range(len(ELEMENTS)) if _dens_d[i] > 0)
        density = molar_mass / molar_vol if molar_vol > 0 else 0
        records.append({
            'Atomic weight mean':               wa_res['Atomic weight mean'],
            'Electronegativity local mismatch': wa_res['Electronegativity local mismatch'],
            'VEC mean':                         wa_res['VEC mean'],
            'Average Ionization Energy':        np.dot(comp, _ie_arr),
            'Average Work Function':            np.dot(comp, _wf_arr),
            'Average IE Difference from Mg':    np.dot(comp, np.abs(_ie_arr - _mg_ie)),
            'Density':                          density,
            'Average Melting Point':            np.dot(comp, _mp_arr),
            'Average Boiling Point':            np.dot(comp, _bp_arr),
            'Average Fermi Energy':             np.dot(comp, _fe_arr),
            'Average Atomic Radius':            np.dot(comp, _ar_arr),
            'Average Surface Energy':           np.dot(comp, _se_arr),
            'Average Adsorption Energy':        np.dot(comp, _ads_arr),
        })
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# 2. 文本标准化
# ══════════════════════════════════════════════════════════════
def _std_immersion(t):
    t = str(t).strip().lower()
    for pat, fn in [
        (r'(\d+(?:\.\d+)?)\s*h(?:ours?)?(?:\s|$|,)', lambda m: f"{float(m.group(1)):.0f} hours"),
        (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?(?:\s|$|,)',  lambda m: f"{int(float(m.group(1))*24)} hours"),
        (r'(\d+(?:\.\d+)?)\s*min(?:utes?)?(?:\s|$|,)',lambda m: f"{float(m.group(1)):.0f} min"),
    ]:
        m = re.search(pat, t)
        if m: return fn(m)
    return t

def _std_ht_proc(t):
    t = str(t).strip().lower()
    for pat, r in [
        (r'\bt4\b','solution-treated'),(r'\bt6\b','solution-treated+aging'),
        (r'solution\s*treat','solution-treated'),(r'anneal','annealed'),
        (r'stress[\s-]?relief','stress-relief treatment'),(r'aging|aged','aging'),
        (r'no\s*heat\s*treat|as[\s-]?cast|without','no heat treatment')]:
        if re.search(pat, t): return r
    return t

def _std_ht_temp(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I):
        return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', t)
    return '+'.join([f"{float(x):.0f}°C" for x in m]) if m else t

def _std_ht_time(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I):
        return 'No heat treatment time'
    def _n(s):
        s = str(s).strip().lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*h', s)
        if m: return f"{float(m.group(1)):.0f} hours"
        m = re.search(r'(\d+(?:\.\d+)?)\s*min', s)
        return f"{float(m.group(1)):.0f} min" if m else s
    return '+'.join([_n(p) for p in re.split(r'[+,;/]', t) if p.strip()])

def _std_deform(t):
    t = str(t).strip().lower()
    for pat, r in [
        (r'cast(?:ed|ing)?|as[\s-]?cast','cast'),(r'roll(?:ed|ing)?','rolling'),
        (r'extrud(?:ed|ing)|extrusion','extruded'),(r'forg(?:ed|ing)?','forging'),
        (r'ECAP|equal\s*channel','ECAP'),(r'HPT','HPT'),(r'FSP','FSP')]:
        if re.search(pat, t, re.I): return r
    return t

def apply_std(df):
    for col, fn in [
        ('Immersion Time',_std_immersion),('Heat Treatment Process',_std_ht_proc),
        ('Heat Treatment Temperature',_std_ht_temp),
        ('Heat Treatment Time',_std_ht_time),('Deformation Process',_std_deform)]:
        if col in df.columns: df[col] = df[col].apply(fn)
    return df


# ══════════════════════════════════════════════════════════════
# 3. 数据加载
# ══════════════════════════════════════════════════════════════
def load_data(path=DATA_PATH):
    df = pd.read_excel(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            mv = df[col].mode()
            df[col] = df[col].fillna(mv[0] if not mv.empty else 'unknown')
        else:
            try: df[col] = df[col].fillna(df[col].median())
            except: pass
    df = df[df[TARGET_COL] <= 30]
    df = df[df[TARGET_COL] <= df[TARGET_COL].mean() + 2.5*df[TARGET_COL].std()]
    df = df[df['Test method'] == 'pw'].copy()
    df = apply_std(df)
    num_cols = df.columns[0:29].tolist()
    num_cols = [c for c in num_cols if df[c].dtype != 'object'
                and c not in [TARGET_COL, DOI_COL]]
    df = df.drop(columns=[c for c in num_cols if (df[c]==0).all()], errors='ignore')
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    df = df[df['year'].notna()].reset_index(drop=True)
    return df

def get_num_cols(df):
    exclude = set(CATEGORICAL_COLUMNS + [TARGET_COL, DOI_COL, 'year'])
    return [c for c in df.columns if df[c].dtype != 'object' and c not in exclude]


# ══════════════════════════════════════════════════════════════
# 4. WAE 定义 & 加载
# ══════════════════════════════════════════════════════════════
class _CBN(nn.Module):
    def forward(self, x):
        if x.size(0) == 1:
            x = x.repeat(2, 1)
            m = x.mean(0, keepdim=True); v = x.var(0, keepdim=True, unbiased=False)
            return ((x - m) / (v + 1e-5).sqrt())[0:1]
        m = x.mean(0, keepdim=True); v = x.var(0, keepdim=True, unbiased=False)
        return (x - m) / (v + 1e-5).sqrt()

class _Softmax(nn.Module):
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        x = F.softmax(x, dim=1)
        return x / x.sum(dim=1, keepdim=True)

class WAE(nn.Module):
    def __init__(self, input_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01),
            nn.Linear(64, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0),
            nn.Linear(128, input_size),
            _Softmax())
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        z = self.encoder(x)
        return self.decoder(z), z

def _replace_bn(m):
    for name, child in m.named_children():
        if isinstance(child, nn.BatchNorm1d): setattr(m, name, _CBN())
        else: _replace_bn(child)

def load_wae(path=WAE_WEIGHTS):
    wae = WAE(input_size=16)
    if os.path.exists(path):
        wae.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        print(f"  WAE weights loaded from {path}")
    else:
        print(f"  [警告] WAE weights not found at {path}，使用随机初始化")
    _replace_bn(wae.encoder); _replace_bn(wae.decoder)
    wae.eval().to(device)
    return wae


# ══════════════════════════════════════════════════════════════
# 5. CombinedNet
# ══════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════
# 6. 工具函数
# ══════════════════════════════════════════════════════════════
def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); random.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_comp(comp):
    comp = np.clip(comp, 0, None)
    s = comp.sum()
    return comp / s if s > 1e-10 else comp

_aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])

def load_pretrained_model():
    """
    加载 year_split 训练好的预测模型和配套 scaler/encoder。
    返回 (model, sX, sy, les, cat_info, num_cols)
    """
    import pickle
    with open(PRETRAINED_SX,      'rb') as f: sX       = pickle.load(f)
    with open(PRETRAINED_SY,      'rb') as f: sy       = pickle.load(f)
    with open(PRETRAINED_LES,     'rb') as f: les      = pickle.load(f)
    with open(PRETRAINED_CATINFO, 'rb') as f: cat_info = pickle.load(f)
    with open(PRETRAINED_NUMCOLS, 'rb') as f: num_cols = pickle.load(f)
    model = CombinedNet(len(num_cols), cat_info,
                        hidden_size=int(BEST_PARAMS['hidden_size']),
                        dropout_rate=float(BEST_PARAMS['dropout_rate'])).to(device)
    model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location=device, weights_only=True))
    model.eval()
    print(f"  预训练模型已加载: {PRETRAINED_MODEL}")
    return model, sX, sy, les, cat_info, num_cols


def atomic_to_wt(comp):
    w = comp * _aw; return w / (w.sum() + 1e-10)

def is_valid(comp, mg_lo=0.90, mg_hi=0.99):
    comp = np.clip(comp, 0, None)
    if comp.sum() < 1e-10: return False
    comp = comp / comp.sum()
    wt = atomic_to_wt(comp)
    if wt[0] < mg_lo or wt[0] > mg_hi: return False
    non_mg = np.sort(wt[1:])[::-1]
    if len(non_mg) > 0 and non_mg[0] > 0.03: return False
    if len(non_mg) > 1 and non_mg[1:].sum() > 0.01: return False
    return True


# ══════════════════════════════════════════════════════════════
# 7. 训练函数
# ══════════════════════════════════════════════════════════════
def build_encoders(full_df):
    """在全量数据上fit LabelEncoder，保证测试集类别不越界"""
    les, cat_info = {}, {}
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder().fit(full_df[col].astype(str))
        les[col] = le; cat_info[col] = len(le.classes_)
    return les, cat_info

def encode_features(df, sX, les, num_cols):
    X_num = sX.transform(df[num_cols].values.astype(float))
    X_cat = np.column_stack([les[c].transform(df[c].astype(str))
                              for c in CATEGORICAL_COLUMNS])
    return X_num, X_cat

def train_model(train_df, test_df, num_cols, les, cat_info, params):
    """
    完全对齐 year_split.py 的 train_nn 逻辑：
    - early stopping 用测试集 loss
    - StepLR step_size=300, gamma=0.5
    - 不在内部重置seed，保持trial内随机性自然演进
    """
    sX = RobustScaler().fit(train_df[num_cols].values.astype(float))
    sy = RobustScaler().fit(train_df[TARGET_COL].values.reshape(-1,1))

    X_tr_num, X_tr_cat = encode_features(train_df, sX, les, num_cols)
    X_te_num, X_te_cat = encode_features(test_df,  sX, les, num_cols)
    y_tr = sy.transform(train_df[TARGET_COL].values.reshape(-1,1)).flatten()
    y_te = sy.transform(test_df[TARGET_COL].values.reshape(-1,1)).flatten()

    X_tr_t  = torch.FloatTensor(X_tr_num).to(device)
    y_tr_t  = torch.FloatTensor(y_tr.reshape(-1,1)).to(device)
    Xc_tr_t = torch.LongTensor(X_tr_cat).to(device)
    X_te_t  = torch.FloatTensor(X_te_num).to(device)
    y_te_t  = torch.FloatTensor(y_te.reshape(-1,1)).to(device)
    Xc_te_t = torch.LongTensor(X_te_cat).to(device)

    model = CombinedNet(X_tr_num.shape[1], cat_info,
                        hidden_size=int(params['hidden_size']),
                        dropout_rate=float(params['dropout_rate'])).to(device)
    crit  = nn.HuberLoss(delta=1.0)
    opt   = torch.optim.AdamW(model.parameters(),
                               lr=float(params['learning_rate']), weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=300, gamma=0.5)
    loader= DataLoader(TensorDataset(X_tr_t, Xc_tr_t, y_tr_t),
                       batch_size=32, shuffle=True, num_workers=0)

    best_loss, best_state, pat = float('inf'), None, 0
    for _ in range(TRAIN_EPOCHS):
        model.train()
        for Xn, Xc, yb in loader:
            opt.zero_grad()
            crit(model(Xn, Xc), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step()
        # early stopping 用测试集 loss（对齐 year_split.py）
        model.eval()
        with torch.no_grad():
            te_loss = crit(model(X_te_t, Xc_te_t), y_te_t).item()
        if te_loss < best_loss:
            best_loss  = te_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
        if pat >= TRAIN_PATIENCE: break

    model.load_state_dict(best_state)
    model.eval()
    return model, sX, sy


# ══════════════════════════════════════════════════════════════
# 8. 预测（标准 + MC Dropout）
# ══════════════════════════════════════════════════════════════
def predict_batch(model, X_num, X_cat, sy):
    model.eval()
    with torch.no_grad():
        p = model(torch.FloatTensor(X_num).to(device),
                  torch.LongTensor(X_cat).to(device)).cpu().numpy()
    return sy.inverse_transform(p).flatten()

def predict_mc(model, X_num, X_cat, sy, n=MC_SAMPLES):
    """MC Dropout：推理时保持dropout开启，多次前向传播"""
    model.train()  # 保持dropout激活
    Xn = torch.FloatTensor(X_num).to(device)
    Xc = torch.LongTensor(X_cat).to(device)
    preds = []
    with torch.no_grad():
        for _ in range(n):
            preds.append(sy.inverse_transform(
                model(Xn, Xc).cpu().numpy()).flatten())
    model.eval()
    preds = np.stack(preds)   # (n, N_cand)
    return preds.mean(0), preds.std(0)


# ══════════════════════════════════════════════════════════════
# 9. WAE+MCMC 候选成分生成
# ══════════════════════════════════════════════════════════════
def generate_wae_mcmc(wae, train_comp_norm, n=N_CANDIDATES, n_steps=3000):
    wae.eval()
    with torch.no_grad():
        z_all = wae.encoder(torch.FloatTensor(train_comp_norm).to(device))
        z_mean = z_all.mean(0).cpu().numpy()

    samples, cur_z = [], z_mean.copy()
    with torch.no_grad():
        cur_comp = wae.decoder(
            torch.FloatTensor(cur_z).unsqueeze(0).to(device))[0].cpu().numpy().flatten()
    cur_valid = is_valid(cur_comp)
    T = 1.0; accept = 0

    for step in range(n_steps):
        r = np.random.rand()
        if r < 0.80:
            prop_z = cur_z + np.random.randn(7) * 0.3
        elif r < 0.95:
            idx = np.random.randint(len(train_comp_norm))
            with torch.no_grad():
                zr = wae.encoder(
                    torch.FloatTensor(train_comp_norm[[idx]]).to(device))
            prop_z = zr.cpu().numpy().flatten() + np.random.randn(7) * 0.5
        else:
            prop_z = np.random.randn(7)

        with torch.no_grad():
            prop_comp = wae.decoder(
                torch.FloatTensor(prop_z).unsqueeze(0).to(device))[0].cpu().numpy().flatten()
        prop_valid = is_valid(prop_comp)

        if prop_valid and (not cur_valid or np.random.rand() < min(1.0, 1.0/T)):
            cur_z, cur_comp, cur_valid = prop_z, prop_comp, prop_valid
            accept += 1

        if prop_valid:
            samples.append(normalize_comp(cur_comp.copy()))

        if (step+1) % 200 == 0:
            ar = accept / 200
            T = T*1.1 if ar < 0.2 else (T*0.9 if ar > 0.5 else T)
            accept = 0
        if len(samples) >= n: break

    while len(samples) < n:
        samples.append(normalize_comp(
            train_comp_norm[np.random.randint(len(train_comp_norm))].copy()))
    return np.array(samples[:n])


# ══════════════════════════════════════════════════════════════
# 10. 候选成分 → 完整特征（描述符 + 文本匹配）
# ══════════════════════════════════════════════════════════════
def build_candidate_df(cand_comps, train_df, num_cols):
    """
    输入: cand_comps (N, 16) 原子分数
    输出: DataFrame，包含完整数值特征（重新计算）和文本特征（最近邻匹配）
    """
    # 计算13个物理描述符
    desc_df = compute_descriptors(cand_comps)

    # 从训练集找最近邻，复制文本特征
    train_comp = train_df[ELEMENTS].values.astype(float)
    train_comp_norm = train_comp / (train_comp.sum(axis=1, keepdims=True) + 1e-10)

    rows = []
    for i, cand in enumerate(cand_comps):
        # 加权Euclidean距离（主合金元素权重3.0）
        weights = np.ones(len(ELEMENTS))
        non_mg = cand[1:]; main_i = np.argmax(non_mg) + 1
        weights[main_i] = 3.0
        dists = np.sqrt(((train_comp_norm - cand)**2 * weights).sum(axis=1))
        ref = train_df.iloc[np.argmin(dists)]

        row = {e: cand[i_e] for i_e, e in enumerate(ELEMENTS)}
        # 物理描述符：用重新计算的值
        for col in DESCRIPTOR_COLS:
            row[col] = desc_df.iloc[i][col]
        # 文本特征：从最近邻复制
        for col in CATEGORICAL_COLUMNS:
            row[col] = ref[col]
        row[TARGET_COL] = 0.0
        rows.append(row)

    cand_df = pd.DataFrame(rows)
    # 确保列顺序与训练集一致
    for col in num_cols:
        if col not in cand_df.columns:
            cand_df[col] = train_df[col].median()
    return cand_df



# ══════════════════════════════════════════════════════════════
# 11. 单次主动学习模拟
# ══════════════════════════════════════════════════════════════
def run_trial(train_init, test_df, full_df,
              num_cols, les, cat_info, params,
              acq_strategy, wae, trial_seed=42):
    """
    一次完整主动学习模拟。
    acq_strategy: 'exploitation' | 'ucb' | 'ei' | 'random'
    WAE+MCMC生成候选 → 预测CR → 采集策略选点 → 预测CR作为实验结果加入训练集
    返回 history dict（每轮的test_r2, test_rmse, best_cr, train_size）
    """
    set_seed(trial_seed)
    current_train = train_init.copy()

    # ── 预计算 baseline_novelty（训练集内部 leave-one-out 平均NN距离）──
    # 和 generation_quality.py 保持一致：归一化后 >1.0 = 比训练集本身更新颖
    _init_comp = train_init[ELEMENTS].values.astype(float)
    _init_comp = _init_comp / (_init_comp.sum(axis=1, keepdims=True) + 1e-10)
    _init_wt_base = np.array([atomic_to_wt(c) for c in _init_comp])[:, 1:] * 100
    _loo_dists = [
        float(np.linalg.norm(np.delete(_init_wt_base, i, 0) - _init_wt_base[i], axis=1).min())
        for i in range(len(_init_wt_base))
    ]
    baseline_novelty = float(np.mean(_loo_dists)) if _loo_dists else 1.0

    # 固定测试集编码（不变）
    sX_test_ref = RobustScaler().fit(train_init[num_cols].values.astype(float))
    X_te_num_raw = test_df[num_cols].values.astype(float)
    X_te_cat = np.column_stack([les[c].transform(test_df[c].astype(str))
                                 for c in CATEGORICAL_COLUMNS])
    y_te = test_df[TARGET_COL].values

    history = {
        'iteration':  [0],
        'test_r2':    [],
        'test_rmse':  [],
        'best_cr':    [current_train[TARGET_COL].min()],
        'train_size': [len(current_train)],
        'diversity':  [0.0],  # iteration=0无选点，用0占位
    }

    for it in range(1, N_ITERATIONS + 1):
        # ── 训练模型 ──────────────────────────────────────────
        # 第1轮：直接加载 year_split 预训练权重，不重新训练
        # 后续轮：用累积训练集重新训练（已包含新实验数据）
        if it == 1:
            model, sX, sy, _, _, _ = load_pretrained_model()
        else:
            model, sX, sy = train_model(current_train, test_df, num_cols, les, cat_info, params)

        # ── 在固定测试集上评估 ─────────────────────────────────
        X_te_num_s = sX.transform(X_te_num_raw)
        pred_te = predict_batch(model, X_te_num_s, X_te_cat, sy)
        r2   = r2_score(y_te, pred_te)
        rmse = np.sqrt(mean_squared_error(y_te, pred_te))

        # ── WAE+MCMC生成候选成分 ───────────────────────────────
        train_comp = current_train[ELEMENTS].values.astype(float)
        train_comp_norm = train_comp / (train_comp.sum(axis=1, keepdims=True) + 1e-10)
        cands = generate_wae_mcmc(wae, train_comp_norm, n=N_CANDIDATES)

        # ── 计算候选成分完整特征 ────────────────────────────────
        cand_df = build_candidate_df(cands, current_train, num_cols)
        X_cand_num_s = sX.transform(cand_df[num_cols].values.astype(float))
        X_cand_cat = np.column_stack([les[c].transform(cand_df[c].astype(str))
                                       for c in CATEGORICAL_COLUMNS])

        # ── 采集策略打分 ───────────────────────────────────────
        if acq_strategy == 'random':
            scores = np.random.rand(len(cands))
            selected_idx = np.random.choice(len(cands), BATCH_SIZE, replace=False)
        else:
            mu, sigma = predict_mc(model, X_cand_num_s, X_cand_cat, sy)
            if acq_strategy == 'exploitation':
                scores = mu
                selected_idx = np.argsort(scores)[:BATCH_SIZE]
            elif acq_strategy == 'ucb':
                scores = mu - UCB_KAPPA * sigma
                selected_idx = np.argsort(scores)[:BATCH_SIZE]
            elif acq_strategy == 'ei':
                best_so_far = current_train[TARGET_COL].min()
                # EI for minimization: E[max(best - y, 0)]
                z = (best_so_far - mu) / (sigma + 1e-9)
                ei = (best_so_far - mu) * scipy_norm.cdf(z) + sigma * scipy_norm.pdf(z)
                ei[sigma < 1e-10] = 0.0
                scores = -ei  # 负号：argsort取最小 = EI最大
                selected_idx = np.argsort(scores)[:BATCH_SIZE]

        # ── 把选出的候选成分+预测CR加入训练集 ────────────────────
        cr_lower = current_train[TARGET_COL].min() * 0.5
        if acq_strategy == 'random':
            mu_all, _ = predict_mc(model, X_cand_num_s, X_cand_cat, sy)
            selected_cr = np.maximum(mu_all[selected_idx], cr_lower)
        else:
            selected_cr = np.maximum(mu[selected_idx], cr_lower)

        new_rows = []
        for i, sidx in enumerate(selected_idx):
            row = cand_df.iloc[sidx].copy()
            row[TARGET_COL] = float(selected_cr[i])
            new_rows.append(row)

        new_df = pd.DataFrame(new_rows)
        for col in current_train.columns:
            if col not in new_df.columns:
                new_df[col] = current_train[col].iloc[0]
        current_train = pd.concat(
            [current_train, new_df[current_train.columns]],
            ignore_index=True)

        # ── 累积Novelty（归一化）：历轮加入成分到初始训练集的平均NN距离 ──
        # 除以 baseline_novelty（训练集内部LOO平均NN距离）归一化
        # >1.0 = 选点比训练集内部更新颖；与 generation_quality.py 体系一致
        n_added = it * BATCH_SIZE
        if n_added > 0:
            added_comp = current_train[ELEMENTS].values[-n_added:].astype(float)
            added_comp = added_comp / (added_comp.sum(axis=1, keepdims=True) + 1e-10)
            added_wt   = np.array([atomic_to_wt(c) for c in added_comp])[:, 1:] * 100
            nn_dists   = [float(np.linalg.norm(_init_wt_base - c, axis=1).min())
                          for c in added_wt]
            diversity  = float(np.mean(nn_dists)) / (baseline_novelty + 1e-10)
        else:
            diversity  = 0.0

        # ── 记录本轮选点的 wt% 成分（供 t-SNE 可视化）────────────
        n_added_this = BATCH_SIZE
        sel_comp = current_train[ELEMENTS].values[-n_added_this:].astype(float)
        sel_comp = sel_comp / (sel_comp.sum(axis=1, keepdims=True) + 1e-10)
        sel_wt   = np.array([atomic_to_wt(c) for c in sel_comp])[:, 1:] * 100
        history.setdefault('selected_wt', []).append(sel_wt)   # list of (BATCH, 15)
        history.setdefault('selected_it', []).append(it)

        history['iteration'].append(it)
        history['test_r2'].append(r2)
        history['test_rmse'].append(rmse)
        history['best_cr'].append(current_train[TARGET_COL].min())
        history['train_size'].append(len(current_train))
        history['diversity'].append(diversity)

    # 补充iteration=0的test指标（用year_split预训练模型，与第1轮起点一致）
    model0, sX0, sy0, _, _, _ = load_pretrained_model()
    X_te0 = sX0.transform(X_te_num_raw)
    pred0 = predict_batch(model0, X_te0, X_te_cat, sy0)
    history['test_r2']  = [r2_score(y_te, pred0)] + history['test_r2']
    history['test_rmse']= [np.sqrt(mean_squared_error(y_te, pred0))] + history['test_rmse']

    return history


# ══════════════════════════════════════════════════════════════
# 13. 运行全部策略
# ══════════════════════════════════════════════════════════════
def run_ablation(df, n_trials=N_TRIALS, out_dir='al_ablation_results'):
    os.makedirs(out_dir, exist_ok=True)

    # ── 数据划分 ──────────────────────────────────────────────
    sim_df  = df[df['year'] < 2025].copy().reset_index(drop=True)
    test_df = df[df['year'] == 2025].copy().reset_index(drop=True)
    print(f"\nSimulation data (year<2025): {len(sim_df)} samples")
    print(f"Fixed test set  (year=2025): {len(test_df)} samples")

    train_init = sim_df.copy().reset_index(drop=True)

    print(f"Initial training set: {len(train_init)} samples "
          f"| CR range [{train_init[TARGET_COL].min():.3f}, "
          f"{train_init[TARGET_COL].max():.3f}]")

    # 从 year_split 预训练文件加载，保证与预训练模型完全一致
    import pickle
    with open(PRETRAINED_NUMCOLS, 'rb') as f: num_cols = pickle.load(f)
    with open(PRETRAINED_LES,     'rb') as f: les      = pickle.load(f)
    with open(PRETRAINED_CATINFO, 'rb') as f: cat_info = pickle.load(f)
    print(f"Numerical features ({len(num_cols)}): {num_cols[:5]}...")

    # 加载WAE
    wae = load_wae(WAE_WEIGHTS)

    # ── 策略列表 ──────────────────────────────────────────────
    strategies = [
        ('exploitation', 'Exploitation (ours)', '#E64B35', '-'),
        ('ucb',          'UCB (MC Dropout)',    '#4DBBD5', '--'),
        ('ei',           'EI (MC Dropout)',     '#00A087', '-.'),
        ('random',       'Random',              '#8491B4', ':'),
    ]

    all_histories = {}
    for acq, label, color, ls in strategies:
        print(f"\n{'─'*55}\nStrategy: {label}")
        trial_hs = []
        for t in range(n_trials):
            seed = 42 + t * 137
            print(f"  Trial {t+1}/{n_trials} (seed={seed})", end='', flush=True)
            h = run_trial(train_init, test_df, df,
                          num_cols, les, cat_info, BEST_PARAMS,
                          acq, wae, trial_seed=seed)
            trial_hs.append(h)
            print(f"  → final test R²={h['test_r2'][-1]:.3f}, "
                  f"best CR={h['best_cr'][-1]:.4f}")
        all_histories[acq] = {
            'label': label, 'color': color, 'ls': ls, 'trials': trial_hs}

        # 保存CSV
        rows = []
        for t, h in enumerate(trial_hs):
            for i, it in enumerate(h['iteration']):
                rows.append({'trial': t+1, 'iteration': it,
                              'test_r2':   h['test_r2'][i],
                              'test_rmse': h['test_rmse'][i],
                              'best_cr':   h['best_cr'][i],
                              'train_size':h['train_size'][i],
                              'diversity': h['diversity'][i]})
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f'{acq}.csv'), index=False)

    # ── 汇总表 ───────────────────────────────────────────────
    summary = []
    for acq, info in all_histories.items():
        trials = info['trials']
        final_r2   = [h['test_r2'][-1]   for h in trials]
        final_rmse = [h['test_rmse'][-1] for h in trials]
        final_cr   = [h['best_cr'][-1]   for h in trials]
        summary.append({
            'strategy':        acq,
            'label':           info['label'],
            'final_r2_mean':   np.mean(final_r2),
            'final_r2_std':    np.std(final_r2),
            'final_rmse_mean': np.mean(final_rmse),
            'final_rmse_std':  np.std(final_rmse),
            'best_cr_mean':    np.mean(final_cr),
            'best_cr_std':     np.std(final_cr),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    print(f"\n{'═'*55}\nSUMMARY")
    print(summary_df[['label','final_r2_mean','final_r2_std',
                       'best_cr_mean','best_cr_std']].to_string(index=False))

    plot_results(all_histories, out_dir)
    plot_tsne(all_histories, train_init, out_dir)

    # ── 保存结果供可视化脚本复用 ─────────────────────────────
    import pickle
    pkl_path = os.path.join(out_dir, 'all_histories.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump({'all_histories': all_histories,
                     'train_init_elements': train_init[ELEMENTS].values.astype(float),
                     'summary': summary_df.to_dict()}, f)
    print(f"Results saved → {pkl_path}")

    return all_histories, summary_df


# ══════════════════════════════════════════════════════════════
# 14. 可视化（参照 Xue 2016 Nat.Commun. Fig.3a / Lookman 2019 Fig.6）
# ══════════════════════════════════════════════════════════════
def plot_results(all_histories, out_dir):
    """
    主图：best-so-far CR随iteration变化，errorbar风格（均值±std）
    辅图：test R² 和 test RMSE 随iteration变化
    均参照 Xue 2016 / Lookman 2019 的图表规范
    """
    matplotlib.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.size':         7,
        'axes.linewidth':    0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size':  3,
        'ytick.major.size':  3,
        'pdf.fonttype':      42,
        'svg.fonttype':      'none',
    })

    # ── 收集各策略各iteration的数据矩阵 ───────────────────────
    max_it = max(
        h['iteration'][-1]
        for info in all_histories.values()
        for h in info['trials']
    )
    iters = np.arange(0, max_it + 1)

    def get_mat(info, key):
        trials = info['trials']
        return np.array([
            np.interp(iters, h['iteration'], h[key])
            for h in trials
        ])

    # ═══════════════════════════════════════════════════════
    # 图1（主图）：best-so-far CR，单独一张，errorbar风格
    # 参照 Xue 2016 Fig.3a
    # ═══════════════════════════════════════════════════════
    fig1, ax = plt.subplots(figsize=(8.8/2.54, 7.0/2.54))

    # 初始训练集最低CR基线（iteration=0时所有策略相同）
    baseline_cr = None
    for info in all_histories.values():
        baseline_cr = info['trials'][0]['best_cr'][0]
        break

    # 画基线虚线（对应 Xue 2016 中的 "best in original set"）
    ax.axhline(y=baseline_cr, color='gray', ls='--', lw=0.8,
               alpha=0.7, label=f'Initial best ({baseline_cr:.3f})')

    markers = {'exploitation': 'o', 'ucb': 's', 'ei': '^', 'random': 'D'}

    for acq, info in all_histories.items():
        mat    = get_mat(info, 'best_cr')
        mu     = mat.mean(0)
        std    = mat.std(0)
        label  = info['label']
        color  = info['color']
        ls     = info['ls']
        marker = markers[acq]

        # 误差棒和连线都在同一x位置，不做偏移
        ax.errorbar(iters, mu, yerr=std,
                    fmt=ls, color=color, lw=1.2,
                    marker=marker, ms=4, markerfacecolor='white',
                    markeredgewidth=0.8, elinewidth=0.8,
                    capsize=2, capthick=0.8,
                    label=label, zorder=3)

    ax.set_xlabel('Number of iterations', fontsize=7)
    ax.set_ylabel('Best-so-far corrosion rate (mm yr$^{-1}$)', fontsize=7)
    ax.set_xlim(-0.5, max_it + 0.5)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5.5, frameon=False, loc='lower left',
              handlelength=2.0, handletextpad=0.5)
    fig1.tight_layout(pad=0.5)

    for ext in ('png', 'svg'):
        fig1.savefig(os.path.join(out_dir, f'al_best_cr.{ext}'),
                     dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)

    # ═══════════════════════════════════════════════════════
    # 图2（辅图）：test R² 和 RMSE，两列并排
    # 参照 Lookman 2019 / 论文 Fig.2d 风格
    # ═══════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(
        1, 3, figsize=(18.0/2.54, 5.5/2.54),
        gridspec_kw={'wspace': 0.40})
    ax_r2, ax_rmse, ax_div = axes2

    for acq, info in all_histories.items():
        color = info['color']
        ls    = info['ls']
        label = info['label']

        for ax, key, ylabel in [
            (ax_r2,   'test_r2',   'Test R²'),
            (ax_rmse, 'test_rmse', 'Test RMSE (mm yr$^{-1}$)'),
        ]:
            mat = get_mat(info, key)
            mu  = mat.mean(0)
            std = mat.std(0)
            ax.plot(iters, mu, color=color, ls=ls, lw=1.2, label=label)
            ax.fill_between(iters, mu - std, mu + std,
                            color=color, alpha=0.10)

        # diversity：从iteration=1开始（iteration=0是0占位）
        div_mat = get_mat(info, 'diversity')
        mu_div  = div_mat.mean(0)[1:]   # 去掉iteration=0
        std_div = div_mat.std(0)[1:]
        iters_div = iters[1:]
        ax_div.plot(iters_div, mu_div, color=color, ls=ls, lw=1.2, label=label)
        ax_div.fill_between(iters_div, mu_div - std_div, mu_div + std_div,
                             color=color, alpha=0.10)

    for ax, ylabel, panel in [
        (ax_r2,   'Test R²',                              'a'),
        (ax_rmse, 'Test RMSE (mm yr$^{-1}$)',             'b'),
        (ax_div,  'Cumulative novelty\n(mean NN dist, wt%)', 'c'),
    ]:
        ax.set_xlabel('Number of iterations', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=7)
        ax.set_xlim(left=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=6)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=5.0, frameon=False, handlelength=1.5)
        ax.text(-0.18, 1.08, panel, transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    fig2.tight_layout(pad=0.5)
    for ext in ('png', 'svg'):
        fig2.savefig(os.path.join(out_dir, f'al_model_perf.{ext}'),
                     dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"\nFigures saved → {out_dir}/al_best_cr.png  |  {out_dir}/al_model_perf.png")


# ══════════════════════════════════════════════════════════════
# 15. t-SNE 成分空间可视化
# ══════════════════════════════════════════════════════════════
def plot_tsne(all_histories, train_init, out_dir):
    """
    风格参照 Allen Brain Atlas UMAP 风格：
    - 背景：初始训练集全部成分，灰色散点密度云（小点，低透明度）
    - 前景：各策略在所有trial的选点，按策略着色，透明度随迭代深度增加
    - 布局：1行4列，每列一个策略，共享同一个t-SNE坐标系
    t-SNE 在所有点（训练集+全部策略所有选点）上联合拟合，保证坐标可比
    """
    matplotlib.rcParams.update({
        'font.family':    'sans-serif',
        'font.sans-serif':['Arial', 'DejaVu Sans'],
        'font.size':      7,
        'axes.linewidth': 0.6,
        'pdf.fonttype':   42,
    })

    # ── 1. 收集所有成分点 ─────────────────────────────────────
    # 初始训练集 wt%（非Mg，15维）
    init_comp = train_init[ELEMENTS].values.astype(float)
    init_comp = init_comp / (init_comp.sum(axis=1, keepdims=True) + 1e-10)
    init_wt   = np.array([atomic_to_wt(c) for c in init_comp])[:, 1:] * 100  # (N,15)

    # 各策略各trial选点：聚合所有trial的选点（去重前），记录(strategy, iteration)
    strat_order = list(all_histories.keys())  # 保序

    # 用第一个trial的selected_wt作为代表（多trial取平均意义不大，t-SNE要具体点）
    # 实际上取所有trial，堆叠后一起放进t-SNE
    all_sel_wt   = []   # 所有策略所有trial所有轮次的选点，shape (M,15)
    all_sel_meta = []   # [(strat_idx, iteration), ...]

    for si, acq in enumerate(strat_order):
        info = all_histories[acq]
        for h in info['trials']:
            for it_idx, sel_wt in enumerate(h.get('selected_wt', [])):
                it = h['selected_it'][it_idx]
                for pt in sel_wt:
                    all_sel_wt.append(pt)
                    all_sel_meta.append((si, it))

    all_sel_wt = np.array(all_sel_wt)   # (M,15)

    # 联合矩阵：训练集 + 所有选点
    combined = np.vstack([init_wt, all_sel_wt])   # (N+M, 15)
    n_init   = len(init_wt)

    # ── 2. 拟合 t-SNE ─────────────────────────────────────────
    print("\nFitting t-SNE on combined compositions...", flush=True)
    tsne = TSNE(n_components=2, perplexity=min(30, len(combined)//4),
                random_state=42, max_iter=1000, learning_rate='auto',
                init='pca', n_jobs=1)
    emb = tsne.fit_transform(combined)   # (N+M, 2)

    init_emb = emb[:n_init]              # 训练集嵌入
    sel_emb  = emb[n_init:]              # 选点嵌入

    # ── 3. 画图：1行4列 ───────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(18.0/2.54, 5.0/2.54),
                              gridspec_kw={'wspace': 0.08})

    for si, (acq, ax) in enumerate(zip(strat_order, axes)):
        info  = all_histories[acq]
        color = info['color']
        label = info['label']

        # ── 背景：所有成分的密度云（灰色）──
        ax.scatter(init_emb[:, 0], init_emb[:, 1],
                   s=6, c='#CCCCCC', alpha=0.5, linewidths=0,
                   rasterized=True, zorder=1)

        # ── 前景：本策略所有trial选点 ──
        # 筛出属于本策略的选点
        mask = np.array([m[0] == si for m in all_sel_meta])
        iters_sel = np.array([m[1] for m in all_sel_meta])

        if mask.sum() > 0:
            pts  = sel_emb[mask]
            its  = iters_sel[mask]
            max_it = its.max()

            # 透明度按迭代轮次从浅到深（早期浅，晚期深）
            alphas = 0.25 + 0.65 * (its / max_it)

            # 用 scatter + 逐点透明度（matplotlib scatter不支持per-point alpha，
            # 改用逐iter画，或用RGBA颜色编码）
            import matplotlib.colors as mcolors
            base_rgb = np.array(mcolors.to_rgb(color))

            for it_val in np.unique(its):
                m2  = its == it_val
                alp = float(0.25 + 0.65 * (it_val / max_it))
                rgba = (*base_rgb, alp)
                ax.scatter(pts[m2, 0], pts[m2, 1],
                           s=18, c=[rgba]*m2.sum(), linewidths=0.3,
                           edgecolors='white', zorder=3)

            # 颜色条指示迭代轮次（只在最后一个子图画）
            if si == len(strat_order) - 1:
                sm = plt.cm.ScalarMappable(
                    cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                        'iter', ['#FFFFFF', color]),
                    norm=plt.Normalize(1, max_it))
                sm.set_array([])

        # ── 标题和外框 ──
        ax.set_title(label, fontsize=6.5, pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color('#AAAAAA')

    # 左侧第一个子图加轴标签
    axes[0].set_xlabel('t-SNE 1', fontsize=6)
    axes[0].set_ylabel('t-SNE 2', fontsize=6)

    # 右侧加颜色条（迭代轮次）
    max_it_global = max(
        h['selected_it'][-1]
        for info in all_histories.values()
        for h in info['trials']
        if h.get('selected_it')
    )
    cbar_ax = fig.add_axes([0.92, 0.20, 0.012, 0.60])
    sm_global = plt.cm.ScalarMappable(
        cmap='Greys',
        norm=plt.Normalize(1, max_it_global))
    sm_global.set_array([])
    cbar = fig.colorbar(sm_global, cax=cbar_ax)
    cbar.set_label('Iteration', fontsize=6, labelpad=4)
    cbar.ax.tick_params(labelsize=5)

    fig.subplots_adjust(right=0.90)
    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'al_tsne.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"t-SNE figure saved → {out_dir}/al_tsne.png")


# ══════════════════════════════════════════════════════════════
# 16. 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',     default=DATA_PATH)
    parser.add_argument('--weights',  default=WAE_WEIGHTS)
    parser.add_argument('--n_iter',   type=int, default=N_ITERATIONS)
    parser.add_argument('--n_trials', type=int, default=N_TRIALS)
    parser.add_argument('--n_init',   type=int, default=N_INIT)
    parser.add_argument('--out',      default='al_ablation_results')
    args = parser.parse_args()

    N_ITERATIONS = args.n_iter
    N_TRIALS     = args.n_trials
    N_INIT       = args.n_init
    WAE_WEIGHTS  = args.weights

    print("="*55)
    print("Active Learning Acquisition Strategy Ablation")
    print(f"Device:       {device}")
    print(f"Data:         {args.data}")
    print(f"WAE weights:  {args.weights}")
    print(f"Init size:    {N_INIT} | Iterations: {N_ITERATIONS} | Batch: {BATCH_SIZE}")
    print(f"Trials:       {N_TRIALS} per strategy")
    print(f"MC samples:   {MC_SAMPLES} | UCB κ: {UCB_KAPPA}")
    print(f"Strategies:   Exploitation / UCB / EI / Random")
    print("="*55)

    df = load_data(args.data)
    run_ablation(df, n_trials=N_TRIALS, out_dir=args.out)
    print("\nDone.")