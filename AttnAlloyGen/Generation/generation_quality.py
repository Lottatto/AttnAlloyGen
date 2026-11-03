import os, re, sys, random, warnings, argparse, pickle
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import WenAlloys

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH    = os.path.join(SCRIPT_DIR, 'database_colab_text_new1.xlsx')
WAE_WEIGHTS  = os.path.join(SCRIPT_DIR, 'WAE_model_weights_pw.pth')

# ── 常量 ──────────────────────────────────────────────────────
ELEMENTS = ['Mg','Al','Zn','Li','Y','Mn','Nd','Ce','Ca','Gd','Zr','Sc','Fe','Sn','In','La']
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
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

ATOMIC_WEIGHTS = {e: Element(e).atomic_mass.real for e in ELEMENTS}
_aw = np.array([ATOMIC_WEIGHTS[e] for e in ELEMENTS])

# ── 各元素在α-Mg(hcp)中的最大平衡固溶度（wt%）──────────────────
# 数据来源：Mezbahul-Islam et al. 2014 (J.Materials); ASM Binary Phase Diagrams;
#           Rokhlin 2003; Okamoto 2000 (Desk Handbook)
# 说明：均为共晶/包晶温度附近的最大值（equilibrium solid solubility limit）
#       多元合金中实际固溶度可能更低，此处作为一阶近似（conservative estimate）
#       Fe固溶度极低（<0.002 wt%），视为有害杂质
MG_SOLUBILITY_WT = {
    'Al':  12.9,   # 共晶温度437°C时 11.8 at% = 12.9 wt% (Murray 1995)
    'Zn':   6.2,   # 共晶温度340°C时 6.2 wt% (Massalski 1986)
    'Li':   5.5,   # ~5.5 wt%（高于此值HCP→BCC相变，合金体系性质改变）
    'Y':   12.4,   # 包晶温度566°C时 ~12.4 wt% (Rokhlin 2003)
    'Mn':   2.2,   # 共晶温度651°C时 ~2.2 wt% (ASM)
    'Nd':   3.6,   # 共晶温度552°C时 ~3.6 wt% (Mezbahul-Islam 2014)
    'Ce':   0.74,  # 共晶温度592°C时 ~0.74 wt% (ASM) —— 固溶度较低
    'Ca':   1.35,  # 共晶温度516.5°C时 0.82 at% = 1.35 wt% (ScienceDirect review)
    'Gd':  23.5,   # 共晶温度548°C时 23.5 wt% (de Gruyter 2024)
    'Zr':   0.6,   # 包晶温度654°C时 ~0.6 wt% (Rokhlin 2003)
    'Sc':  24.0,   # 共晶温度 ~24 wt% (PMC 2022 single-phase RE alloys)
    'Fe':   0.002, # 几乎不固溶，有害杂质（Siebel, Zeitschrift für Metallkunde）
    'Sn':  14.5,   # 共晶温度561°C时 3.35 at% = 14.5 wt% (ScienceDirect review)
    'In':  19.4,   # ~19.4 wt% (Okamoto 2000)
    'La':   1.0,   # 共晶温度615°C时 ~1.0 wt% (ASM)
}

N_GEN = 500  # 每种方法生成的候选成分数

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════
# 1. 描述符计算（从原子分数直接计算所有13个数值特征）
# ══════════════════════════════════════════════════════════════
# 所有字典从 dataset.ipynb / add_line.ipynb 提取
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

# 预计算固定数组（加速批量计算）
_ie_arr  = np.array([Element(e).ionization_energies[0] for e in ELEMENTS])
_wf_arr  = np.array([_WF.get(e, 0)   for e in ELEMENTS])
_mp_arr  = np.array([_MP.get(e, 0)   for e in ELEMENTS])
_bp_arr  = np.array([_BP.get(e, 0)   for e in ELEMENTS])
_fe_arr  = np.array([_FE.get(e, 0)   for e in ELEMENTS])
_se_arr  = np.array([_SE.get(e, 0)   for e in ELEMENTS])
_ads_arr = np.array([_ADS.get(e, 0)  for e in ELEMENTS])
_ar_arr  = np.array([Element(e).atomic_radius.real for e in ELEMENTS])
_mg_ie   = Element('Mg').ionization_energies[0]
_dens_aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
_dens_d  = np.array([_DENS.get(e, 1) for e in ELEMENTS])

# WenAlloys featurizer（ΔEN、AWM、VEC用它）
_wa = WenAlloys()


def compute_descriptors(comp_array):
    """
    输入: comp_array shape (N, 16) 原子分数（已归一化）
    输出: DataFrame，列名与 database_colab_text_new1.xlsx 的数值特征完全一致
    每一行对应一个候选成分的完整描述符。
    """
    N = len(comp_array)
    records = []

    for comp in comp_array:
        comp = np.clip(comp, 0, None)
        s = comp.sum()
        if s < 1e-10: comp = np.zeros(len(ELEMENTS)); comp[0] = 1.0
        else: comp = comp / s

        # WenAlloys特征（AWM, ΔEN, VEC）
        formula_parts = [f"{e}{comp[i]:.6f}"
                         for i, e in enumerate(ELEMENTS) if comp[i] > 1e-8]
        c = Composition(''.join(formula_parts))
        wa_res = dict(zip(_wa.feature_labels(), _wa.featurize(c)))

        # 密度（摩尔体积法）
        molar_mass = np.dot(comp, _dens_aw)
        molar_vol  = sum(comp[i] * _dens_aw[i] / _dens_d[i]
                         for i in range(len(ELEMENTS)) if _dens_d[i] > 0)
        density = molar_mass / molar_vol if molar_vol > 0 else 0

        records.append({
            'Atomic weight mean':              wa_res['Atomic weight mean'],
            'Electronegativity local mismatch':wa_res['Electronegativity local mismatch'],
            'VEC mean':                        wa_res['VEC mean'],
            'Average Ionization Energy':       np.dot(comp, _ie_arr),
            'Average Work Function':           np.dot(comp, _wf_arr),
            'Average IE Difference from Mg':   np.dot(comp, np.abs(_ie_arr - _mg_ie)),
            'Density':                         density,
            'Average Melting Point':           np.dot(comp, _mp_arr),
            'Average Boiling Point':           np.dot(comp, _bp_arr),
            'Average Fermi Energy':            np.dot(comp, _fe_arr),
            'Average Atomic Radius':           np.dot(comp, _ar_arr),
            'Average Surface Energy':          np.dot(comp, _se_arr),
            'Average Adsorption Energy':       np.dot(comp, _ads_arr),
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# 2. 数据加载
# ══════════════════════════════════════════════════════════════
def _std_immersion(t):
    t = str(t).strip().lower()
    for p, fn in [(r'(\d+(?:\.\d+)?)\s*h(?:ours?)?', lambda m: f"{float(m.group(1)):.0f} hours"),
                  (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?',  lambda m: f"{int(float(m.group(1))*24)} hours")]:
        m = re.search(p, t)
        if m: return fn(m)
    return t

def _std_ht(t):
    t = str(t).strip().lower()
    for p, r in [(r'\bt4\b','solution-treated'),(r'\bt6\b','solution-treated+aging'),
                 (r'solution\s*treat','solution-treated'),(r'anneal','annealed'),
                 (r'aging|aged','aging'),(r'no\s*heat\s*treat|as[\s-]?cast','no heat treatment')]:
        if re.search(p, t): return r
    return t

def _std_temp(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I): return 'No heat treatment temperature'
    m = re.findall(r'(\d+(?:\.\d+)?)\s*°?\s*[Cc]', t)
    return '+'.join([f"{float(x):.0f}°C" for x in m]) if m else t

def _std_time(t):
    t = str(t).strip()
    if re.search(r'no|none|without|as.cast|n/a', t, re.I): return 'No heat treatment time'
    def _n(s):
        s = str(s).strip().lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*h', s)
        if m: return f"{float(m.group(1)):.0f} hours"
        m = re.search(r'(\d+(?:\.\d+)?)\s*min', s)
        if m: return f"{float(m.group(1)):.0f} min"
        return s
    return '+'.join([_n(p) for p in re.split(r'[+,;/]', t) if p.strip()])

def _std_deform(t):
    t = str(t).strip().lower()
    for p, r in [(r'cast(?:ed|ing)?|as[\s-]?cast','cast'),(r'roll(?:ed|ing)?','rolling'),
                 (r'extrud(?:ed|ing)|extrusion','extruded')]:
        if re.search(p, t, re.I): return r
    return t

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
    for col, fn in [('Immersion Time',_std_immersion),('Heat Treatment Process',_std_ht),
                    ('Heat Treatment Temperature',_std_temp),('Heat Treatment Time',_std_time),
                    ('Deformation Process',_std_deform)]:
        if col in df.columns: df[col] = df[col].apply(fn)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    df = df[df['year'].notna()].reset_index(drop=True)
    return df



# ══════════════════════════════════════════════════════════════
# 3. WAE 定义 & 加载
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
        x = F.softmax(x, dim=1); return x / x.sum(dim=1, keepdim=True)

class WAE(nn.Module):
    def __init__(self, input_size=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 128), nn.BatchNorm1d(128, track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01),
            nn.Linear(128, 64), nn.BatchNorm1d(64, track_running_stats=False),
            nn.LeakyReLU(0.0), nn.Dropout(0.01), nn.Linear(64, 7))
        self.decoder = nn.Sequential(
            nn.Linear(7, 64), nn.BatchNorm1d(64, track_running_stats=False), nn.LeakyReLU(0.0),
            nn.Linear(64, 128), nn.BatchNorm1d(128, track_running_stats=False), nn.LeakyReLU(0.0),
            nn.Linear(128, input_size), _Softmax())
    def forward(self, x):
        if len(x.shape) == 1: x = x.unsqueeze(0)
        z = self.encoder(x); return self.decoder(z), z

def _replace_bn(m):
    for name, child in m.named_children():
        if isinstance(child, nn.BatchNorm1d): setattr(m, name, _CBN())
        else: _replace_bn(child)

def load_wae(path=WAE_WEIGHTS):
    wae = WAE(16)
    if not os.path.exists(path):
        raise FileNotFoundError(f"WAE权重文件不存在：{path}。请先运行train_wae.py或检查文件名。")
    wae.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
    _replace_bn(wae.encoder); _replace_bn(wae.decoder)
    wae.eval().to(device)
    return wae


# ══════════════════════════════════════════════════════════════
# 5. 物理约束
# ══════════════════════════════════════════════════════════════
def atomic_to_wt(comp):
    w = comp * _aw; return w / (w.sum() + 1e-10)

def is_valid(comp, mg_lo=0.90, mg_hi=0.99):
    """硬约束：用于评估指标统计
    Mg wt%: [mg_lo, mg_hi]（默认90-99%），主合金元素 ≤10%，次要元素总和 ≤3%
    mg_lo/mg_hi 供敏感性分析时调整，正常使用保持默认值。
    """
    comp = np.clip(comp, 0, None)
    if comp.sum() < 1e-10: return False
    comp = comp / comp.sum()
    wt = atomic_to_wt(comp)
    if wt[0] < mg_lo or wt[0] > mg_hi: return False
    non_mg = np.sort(wt[1:])[::-1]
    if len(non_mg) > 0 and non_mg[0] > 0.10: return False
    if len(non_mg) > 1 and non_mg[1:].sum() > 0.03: return False
    return True

def post_process_comp(comp, mg_lo=0.90, mg_hi=0.99):
    """
    后处理：把WAE解码出来的成分强制修正到满足物理约束。
    Mg wt%: 90-99%，主合金元素 ≤10%，次要元素总和 ≤3%
    Fe直接归零（在Mg合金里是有害杂质，不作为合金元素添加）
    mg_lo/mg_hi 供敏感性分析时调整约束范围，正常使用保持默认值。
    """
    comp = np.maximum(comp, 0)
    if comp.sum() < 1e-10: return comp

    fe_idx = ELEMENTS.index('Fe')
    comp[fe_idx] = 0.0
    comp = comp / comp.sum()

    wt = atomic_to_wt(comp)
    mg_wt = wt[0]

    # 修正Mg含量：目标非Mg比例 = 0.08（使Mg约达到0.92），与原逻辑完全一致
    if mg_wt < mg_lo:
        target_nonmg = 0.08
        scale = target_nonmg / (1 - mg_wt + 1e-10)
        for i in range(1, len(comp)): comp[i] *= scale
    elif mg_wt > mg_hi:
        target_nonmg = 0.03
        scale = target_nonmg / (1 - mg_wt + 1e-10)
        for i in range(1, len(comp)): comp[i] *= scale

    comp = comp / (comp.sum() + 1e-10)

    wt = atomic_to_wt(comp)
    non_mg_idx = list(range(1, len(ELEMENTS)))
    sorted_idx = sorted(non_mg_idx, key=lambda i: wt[i], reverse=True)

    if wt[sorted_idx[0]] > 0.10:
        scale = 0.10 / (wt[sorted_idx[0]] + 1e-10)
        comp[sorted_idx[0]] *= scale

    secondary_idx = sorted_idx[1:]
    wt = atomic_to_wt(comp / (comp.sum() + 1e-10))
    total_sec = sum(wt[i] for i in secondary_idx)
    if total_sec > 0.03:
        scale = 0.03 / (total_sec + 1e-10)
        for i in secondary_idx: comp[i] *= scale

    comp = comp / (comp.sum() + 1e-10)
    wt = atomic_to_wt(comp)
    comp[wt < 0.005] = 0

    comp = comp / (comp.sum() + 1e-10)
    wt = atomic_to_wt(comp)
    non_mg_wt = [(i, wt[i]) for i in range(1, len(ELEMENTS)) if wt[i] > 0]
    non_mg_wt.sort(key=lambda x: x[1], reverse=True)
    keep_idx = {x[0] for x in non_mg_wt[:2]}  # 保留前3个，允许四元合金
    for i in range(1, len(ELEMENTS)):
        if i not in keep_idx: comp[i] = 0

    comp = comp / (comp.sum() + 1e-10)
    return comp

def score_comp(comp, mg_lo=0.90, mg_hi=0.99):
    """
    软评分函数：用于MCMC接受准则。
    mg_lo/mg_hi 供敏感性分析时调整约束范围，正常使用保持默认值。
    默认值下行为与原版本完全一致。
    """
    comp = np.clip(comp, 0, None)
    if comp.sum() < 1e-10: return 1e-10
    comp = comp / comp.sum()
    wt = atomic_to_wt(comp)
    mg_wt = wt[0]
    non_mg = np.sort(wt[1:])[::-1]
    main_alloy = non_mg[0] if len(non_mg) > 0 else 0
    secondary  = non_mg[1:].sum() if len(non_mg) > 1 else 0

    if secondary > 0.03: return 1e-10
    if mg_wt < 0.80 or mg_wt > 0.999: return 1e-10

    # mg_score：原版hardcode边界，默认值下完全不变
    if mg_wt < mg_lo:
        mg_score = 0.1
    elif mg_wt < mg_lo + 0.02:
        mg_score = 0.4 + (mg_wt - mg_lo) * 30
    elif mg_wt > mg_hi - 0.02:
        mg_score = 1.0 - (mg_wt - (mg_hi - 0.02)) * 90
    else:
        mg_score = 1.0
    mg_score = max(mg_score, 1e-10)

    if main_alloy <= 0.10:
        alloy_score = 1.0
    elif main_alloy <= 0.12:
        alloy_score = 1.0 - (main_alloy - 0.10) / 0.02
    else:
        return 1e-10

    comp_score = mg_score * 0.7 + alloy_score * 0.3

    n_alloy_elems = int(np.sum(wt[1:] >= 0.005))
    variety_score = 1.0 if n_alloy_elems <= 3 else (0.7 if n_alloy_elems == 4
                    else 0.3)
    comp_score = comp_score * 0.7 + variety_score * 0.3

    return float(max(comp_score * (1 + np.random.normal(0, 0.01)), 1e-10))

def normalize_comp(c):
    c = np.clip(c, 0, None); s = c.sum()
    return c/s if s > 1e-10 else c


# ══════════════════════════════════════════════════════════════
# 6. 四种生成方法（统一物理约束）
# ══════════════════════════════════════════════════════════════
def generate_wae_mcmc(wae, train_comp, n=N_GEN, n_steps=8000):
    """
    多起点并行MCMC：从所有valid训练成分各起一条链，
    每条链贡献 n//n_chains 个样本，保证体系覆盖均匀，
    不再只围着WAE潜空间密度高的Ce/Gd区域转。
    """
    wae.eval()
    with torch.no_grad():
        z_all    = wae.encoder(torch.FloatTensor(train_comp).to(device))
        z_all_np = z_all.cpu().numpy()

    valid_mask    = np.array([is_valid(c) for c in train_comp])
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(len(train_comp))

    # ── 决定并行链数和每链采样数 ──────────────────────────────
    # 按KMeans聚类均匀选起点，保证每个成分族都有链覆盖
    N_START = min(15, len(valid_indices))
    km_start = KMeans(n_clusters=N_START, random_state=42, n_init=10)
    km_start.fit(train_comp[valid_indices])
    cluster_labels = km_start.labels_
    # 每个族选离聚类中心最近的训练成分作为起点
    start_indices = []
    for ci in range(N_START):
        members = np.where(cluster_labels == ci)[0]
        center  = km_start.cluster_centers_[ci]
        dists   = np.linalg.norm(train_comp[valid_indices[members]] - center, axis=1)
        start_indices.append(valid_indices[members[np.argmin(dists)]])
    start_indices = np.array(start_indices)

    n_chains          = N_START
    samples_per_chain = max(1, (n + n_chains - 1) // n_chains)
    steps_per_chain   = max(600, n_steps // n_chains)

    print(f"    MCMC: {n_chains} chains × {steps_per_chain} steps, "
          f"{samples_per_chain} samples/chain | valid_train={len(valid_indices)}/{len(train_comp)}")

    all_samples = []

    for chain_i, vi in enumerate(start_indices):
        # 起点：直接用训练集成分（不走WAE编解码，保证score>0）
        cur_c = normalize_comp(train_comp[vi].copy())
        cur_z = z_all_np[vi].copy()
        cur_s = score_comp(cur_c)

        chain_samples = []
        T, n_acc, n_att = 2.0, 0, 0

        for step in range(steps_per_chain):
            n_att += 1
            r = np.random.rand()

            if r < 0.50:
                # 成分空间局部扰动
                sigma = np.random.choice([0.01, 0.02, 0.03, 0.04])
                pc = normalize_comp(cur_c + np.random.randn(len(ELEMENTS)) * sigma)
                if not is_valid(pc):
                    pc = post_process_comp(pc)
                with torch.no_grad():
                    pz = wae.encoder(
                        torch.FloatTensor(pc).unsqueeze(0).to(device)).cpu().numpy().flatten()

            elif r < 0.85:
                # WAE潜空间游走
                pz = cur_z + np.random.randn(7) * 0.35
                with torch.no_grad():
                    pc = wae.decoder(
                        torch.FloatTensor(pz).unsqueeze(0).to(device))[0].cpu().numpy().flatten()
                pc = post_process_comp(pc)

            else:
                # 全局跳转：随机选一个valid训练成分
                idx  = np.random.choice(valid_indices)
                base = train_comp[idx].copy()
                pc   = normalize_comp(base + np.random.randn(len(ELEMENTS)) * 0.01)
                if not is_valid(pc):
                    pc = normalize_comp(base.copy())
                with torch.no_grad():
                    pz = wae.encoder(
                        torch.FloatTensor(pc).unsqueeze(0).to(device)).cpu().numpy().flatten()

            ps = score_comp(pc)

            if cur_s > 1e-9:
                accept_ratio = min(1.0, (ps / cur_s) ** (1.0 / T))
            else:
                accept_ratio = 1.0
            if np.random.rand() < accept_ratio:
                cur_z, cur_c, cur_s = pz, pc, ps
                n_acc += 1

            if is_valid(cur_c):
                chain_samples.append(
                    normalize_comp(post_process_comp(cur_c.copy())))

            # 动态调温
            if (step + 1) % 100 == 0:
                ar = n_acc / max(1, n_att)
                T  = max(T * 1.1 if ar < 0.2 else (T * 0.9 if ar > 0.5 else T), 0.5)
                n_acc = n_att = 0

            if len(chain_samples) >= samples_per_chain:
                break

        # 补足：直接用起点训练成分加小扰动
        while len(chain_samples) < samples_per_chain:
            cand = normalize_comp(train_comp[vi] + np.random.randn(len(ELEMENTS)) * 0.005)
            cand = post_process_comp(cand if is_valid(cand) else normalize_comp(train_comp[vi].copy()))
            chain_samples.append(normalize_comp(cand))

        all_samples.extend(chain_samples[:samples_per_chain])

    # 去重（tolerance=0.001）+ 补足，不影响全局随机状态
    _rng_state     = np.random.get_state()
    _torch_state   = torch.get_rng_state()
    _random_state  = random.getstate()

    all_samples = np.array(all_samples)
    keep = [0]
    for i in range(1, len(all_samples)):
        if not any(np.all(np.abs(all_samples[i] - all_samples[j]) < 0.001) for j in keep):
            keep.append(i)
    all_samples = all_samples[keep]

    # 如果样本超出n，随机打乱后截取
    if len(all_samples) > n:
        idx = np.random.choice(len(all_samples), n, replace=False)
        all_samples = all_samples[idx]
    elif len(all_samples) < n:
        # 补足（加小扰动避免引入重复）
        extra = []
        while len(extra) < n - len(all_samples):
            base = all_samples[np.random.randint(len(all_samples))]
            cand = normalize_comp(base + np.random.randn(len(ELEMENTS)) * 0.005)
            cand = post_process_comp(cand)
            if is_valid(cand):
                extra.append(cand)
        all_samples = np.vstack([all_samples, np.array(extra)])

    # 恢复随机状态，保证后续方法结果不受影响
    np.random.set_state(_rng_state)
    torch.set_rng_state(_torch_state)
    random.setstate(_random_state)

    return all_samples[:n]


def generate_random(train_comp, n=N_GEN):
    valid_mask    = np.array([is_valid(c) for c in train_comp])
    valid_indices = np.where(valid_mask)[0]
    results, attempts = [], 0
    while len(results) < n and attempts < n * 100:
        attempts += 1
        # 优先从valid子集出发扰动，提高命中率
        if len(valid_indices) > 0 and np.random.rand() < 0.7:
            base = train_comp[np.random.choice(valid_indices)].copy()
        else:
            base = train_comp[np.random.randint(len(train_comp))].copy()
        cand = normalize_comp(base + np.random.randn(len(ELEMENTS)) *
                              np.random.choice([0.005, 0.01, 0.02, 0.05]))
        if is_valid(cand): results.append(normalize_comp(post_process_comp(cand)))
    # 补足：从valid子集加小扰动
    while len(results) < n:
        if len(valid_indices) > 0:
            idx  = np.random.choice(valid_indices)
            cand = normalize_comp(
                train_comp[idx] + np.random.randn(len(ELEMENTS)) * 0.005)
            cand = cand if is_valid(cand) else normalize_comp(train_comp[idx].copy())
            results.append(normalize_comp(post_process_comp(cand)))
        else:
            results.append(normalize_comp(
                post_process_comp(train_comp[np.random.randint(len(train_comp))].copy())))
    return np.array(results[:n])


def generate_bayesopt(train_comp, train_cr, n=N_GEN, n_pool=1000):
    from scipy.stats import norm as sp_norm
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_gp = sc.fit_transform(train_comp)
    gp   = GaussianProcessRegressor(kernel=Matern(nu=2.5), alpha=1e-3,
                                     normalize_y=True, n_restarts_optimizer=3)
    gp.fit(X_gp, -train_cr)  # 最大化 → 最小化CR
    pool    = generate_random(train_comp, n=n_pool)
    pool_sc = sc.transform(pool)
    mu, sigma = gp.predict(pool_sc, return_std=True)
    best_y = (-train_cr).max()
    Z  = (mu - best_y) / (sigma + 1e-9)
    ei = (mu - best_y) * sp_norm.cdf(Z) + sigma * sp_norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    top = np.argsort(ei)[::-1][:n]
    return pool[top]


def generate_grid(train_comp, n=N_GEN):
    """
    Grid search：在训练集成分的基础上，对主合金元素含量做系统性网格扫描。
    对每个元素在[0.5%, 5%]范围内取若干个离散值，二元/三元组合，
    过滤满足物理约束的成分，选前n个。
    参考Xue 2016 NiTi SMA的网格化成分空间思路。
    """
    valid_mask    = np.array([is_valid(c) for c in train_comp])
    valid_indices = np.where(valid_mask)[0]

    # 非Mg元素集合（只用训练集里出现过的元素，避免离谱组合）
    wt_matrix = np.array([atomic_to_wt(c) for c in train_comp]) * 100
    # 找到训练集里wt%>0.3%的非Mg元素
    active_elems = []
    for i, e in enumerate(ELEMENTS[1:], 1):
        if wt_matrix[:, i].max() > 0.3:
            active_elems.append(i)

    # 网格值：5个离散点 [0.5, 1.0, 2.0, 3.5, 5.0] wt%
    grid_vals_wt = [0.5, 1.0, 2.0, 3.5, 5.0]

    results = []
    # 二元合金（Mg + 1种元素）
    for ei in active_elems:
        for v1 in grid_vals_wt:
            # 构造wt%成分
            wt = np.zeros(len(ELEMENTS))
            wt[0] = 100 - v1  # Mg
            wt[ei] = v1
            # 转原子分数
            at = wt / np.array([ATOMIC_WEIGHTS[e] for e in ELEMENTS])
            at = at / (at.sum() + 1e-10)
            comp = post_process_comp(at)
            if is_valid(comp):
                results.append(comp)

    # 三元合金（Mg + 2种元素）
    for i1, ei in enumerate(active_elems):
        for ej in active_elems[i1+1:]:
            for v1 in [0.5, 1.0, 2.0]:
                for v2 in [0.5, 1.0]:
                    if v1 + v2 > 8: continue
                    wt = np.zeros(len(ELEMENTS))
                    wt[0] = 100 - v1 - v2
                    wt[ei] = v1; wt[ej] = v2
                    at = wt / np.array([ATOMIC_WEIGHTS[e] for e in ELEMENTS])
                    at = at / (at.sum() + 1e-10)
                    comp = post_process_comp(at)
                    if is_valid(comp):
                        results.append(comp)

    results = np.array(results) if results else np.array([train_comp[0]])

    # 若超出n则随机选；若不足则用Random search补足
    if len(results) >= n:
        idx = np.random.choice(len(results), n, replace=False)
        return results[idx]
    else:
        extra = generate_random(train_comp, n - len(results))
        return np.vstack([results, extra])


def generate_genetic(train_comp, n=N_GEN,
                     pop_size=200, n_gen=20, mutation_rate=0.15):
    """
    遗传算法：用成分约束评分（score_comp）作为fitness，
    不依赖预测模型，纯粹优化成分合理性+稀疏性。
    """
    valid_mask    = np.array([is_valid(c) for c in train_comp])
    valid_indices = np.where(valid_mask)[0]
    base_pool = train_comp[valid_indices] if len(valid_indices) > 0 else train_comp

    pop = []
    attempts = 0
    while len(pop) < pop_size and attempts < pop_size * 50:
        attempts += 1
        base = base_pool[np.random.randint(len(base_pool))].copy()
        cand = post_process_comp(normalize_comp(
            base + np.random.randn(len(ELEMENTS)) * 0.01))
        if is_valid(cand): pop.append(cand)
    while len(pop) < pop_size:
        pop.append(normalize_comp(base_pool[np.random.randint(len(base_pool))].copy()))
    pop = np.array(pop)

    for _ in range(n_gen):
        # fitness = score_comp（成分约束评分），越高越好
        fitness = np.array([score_comp(c) for c in pop])

        new_pop = []
        attempts = 0
        while len(new_pop) < pop_size and attempts < pop_size * 20:
            attempts += 1
            i1, i2 = np.random.randint(pop_size, size=2)
            p1 = pop[i1] if fitness[i1] > fitness[i2] else pop[i2]  # 选score更高的
            i3, i4 = np.random.randint(pop_size, size=2)
            p2 = pop[i3] if fitness[i3] > fitness[i4] else pop[i4]
            mask  = np.random.rand(len(ELEMENTS)) > 0.5
            noise = np.random.randn(len(ELEMENTS)) * 0.008 if np.random.rand() < mutation_rate else 0
            child = post_process_comp(normalize_comp(np.where(mask, p1, p2) + noise))
            if is_valid(child):
                new_pop.append(child)
        if len(new_pop) >= pop_size:
            pop = np.array(new_pop[:pop_size])

    # 按score排序，返回最好的n个
    final_scores = np.array([score_comp(c) for c in pop])
    return pop[np.argsort(final_scores)[::-1][:n]]


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# 8. 质量评估指标
# ══════════════════════════════════════════════════════════════
def evaluate_generation_quality(cands, train_comp, label):
    """
    生成质量评估（纯成分空间，不涉及腐蚀速率预测）

    指标：
    1. validity        有效率（物理约束满足比例，均为1.0，文字说明即可）
    2. diversity_norm  归一化多样性（生成成分pairwise距离 / 训练集内部pairwise距离）
                       >1.0 = 比训练集更分散
    3. novelty_norm    归一化新颖性（生成到训练集最近邻距离 / 训练集内部最近邻距离）
                       >1.0 = 比训练集本身更新
    4. coverage        成分族覆盖率（KMeans 15族）
    5. dist_consistency 分布一致性（MMD，越小=生成分布越接近训练集分布）
                        用于说明WAE学到了真实Mg合金分布
    """
    cands_wt = np.array([atomic_to_wt(c) for c in cands])[:, 1:] * 100
    train_wt = np.array([atomic_to_wt(c) for c in train_comp])[:, 1:] * 100

    # ── 训练集内部基准（只算一次，供diversity和novelty共用）────
    n_tr = len(train_wt)
    # 训练集内部平均pairwise距离（采样加速）
    n_tr_s = min(100, n_tr)
    tr_idx = np.random.choice(n_tr, n_tr_s, replace=False)
    tr_s = train_wt[tr_idx]
    tr_dists = [np.linalg.norm(tr_s[i]-tr_s[j])
                for i in range(n_tr_s) for j in range(i+1, n_tr_s)]
    baseline_diversity = float(np.mean(tr_dists)) if tr_dists else 1.0

    # 训练集内部平均最近邻距离（leave-one-out）
    baseline_novelty = float(np.mean([
        np.linalg.norm(np.delete(train_wt, i, 0) - train_wt[i], axis=1).min()
        for i in range(n_tr)]))

    # (1) 有效率
    validity = float(np.mean([is_valid(c) for c in cands]))

    # (2) 归一化多样性
    n_s = min(200, len(cands))
    idx = np.random.choice(len(cands), n_s, replace=False)
    s   = cands_wt[idx]
    dists = [np.linalg.norm(s[i]-s[j]) for i in range(n_s) for j in range(i+1, n_s)]
    raw_diversity = float(np.mean(dists)) if dists else 0.0
    diversity_norm = raw_diversity / (baseline_diversity + 1e-10)

    # (3) 归一化新颖性
    raw_nov = float(np.mean([np.linalg.norm(train_wt-c, axis=1).min()
                              for c in cands_wt]))
    novelty_norm = raw_nov / (baseline_novelty + 1e-10)

    # (4) Coverage
    N_CL = min(15, n_tr // 5)
    if N_CL >= 2:
        km = KMeans(n_clusters=N_CL, random_state=42, n_init=10)
        km.fit(train_wt)
        coverage = float(len(np.unique(km.predict(cands_wt)))) / N_CL
    else:
        coverage = 1.0

    # (5) 分布一致性：MMD（Maximum Mean Discrepancy）
    # 用RBF核，bandwidth取训练集pairwise距离中位数（median heuristic）
    # MMD越小 = 生成分布越接近训练集分布 = WAE学得越好
    def _mmd(X, Y, bandwidth=None):
        """无偏MMD估计，RBF核"""
        if bandwidth is None:
            # median heuristic
            all_dists = [np.linalg.norm(X[i]-X[j])
                         for i in range(min(50,len(X)))
                         for j in range(i+1, min(50,len(X)))]
            bandwidth = float(np.median(all_dists)) if all_dists else 1.0
        bw2 = 2 * bandwidth**2

        def _rbf(A, B):
            # (N,D) x (M,D) -> (N,M) kernel matrix mean
            diff = A[:, None, :] - B[None, :, :]          # (N,M,D)
            return float(np.exp(-np.sum(diff**2, axis=-1) / bw2).mean())

        n, m = min(200, len(X)), min(200, len(Y))
        Xs = X[np.random.choice(len(X), n, replace=False)]
        Ys = Y[np.random.choice(len(Y), m, replace=False)]
        return _rbf(Xs, Xs) - 2*_rbf(Xs, Ys) + _rbf(Ys, Ys)

    mmd = _mmd(cands_wt, train_wt)
    # 归一化：除以Random search的MMD作为基准（在run_part1里做）
    # 这里先返回raw MMD，后处理时归一化

    print(f"  {label:35s}  "
          f"diversity={diversity_norm:.3f}  novelty={novelty_norm:.3f}  "
          f"coverage={coverage:.2f}  MMD={mmd:.4f}")

    return {'label': label, 'validity': validity,
            'diversity': diversity_norm, 'raw_diversity': raw_diversity,
            'novelty': novelty_norm,
            'coverage': coverage, 'mmd': mmd,
            'cands': cands}


# ══════════════════════════════════════════════════════════════
# 9. 主流程
# ══════════════════════════════════════════════════════════════
def run_part1(n_gen=N_GEN, out_dir='generation_quality_results'):
    global N_GEN
    N_GEN = n_gen
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(42); random.seed(42); torch.manual_seed(42)

    print("\n" + "="*60)
    print("Part 1: Generation Quality Evaluation")
    print(f"  n_gen={n_gen} per method | Device={device}")
    print("="*60)

    # ── 加载数据 ──────────────────────────────────────────────
    df = load_data(DATA_PATH)
    train_df   = df[df['year'] < 2025].copy().reset_index(drop=True)
    train_comp = train_df[ELEMENTS].values.astype(float)
    train_comp = train_comp / (train_comp.sum(axis=1, keepdims=True) + 1e-10)
    train_cr   = train_df[TARGET_COL].values
    print(f"  Train data (year<2025): {len(train_df)} samples")

    # ── 加载WAE ───────────────────────────────────────────────
    print("Loading WAE...")
    wae = load_wae(WAE_WEIGHTS)

    # ── 生成 & 评估 ────────────────────────────────────────────
    print(f"\nGenerating {n_gen} candidates per method and evaluating...")
    print(f"  {'Method':35s}  validity   diversity   novelty  coverage  physical")
    print("  " + "-"*80)

    methods = [
        ('WAE+MCMC (ours)', lambda: generate_wae_mcmc(wae, train_comp, n_gen)),
        ('Random search',   lambda: generate_random(train_comp, n_gen)),
        ('Grid search',     lambda: generate_grid(train_comp, n_gen)),
        ('Bayesian Opt+EI', lambda: generate_bayesopt(train_comp, train_cr, n_gen)),
        ('Genetic Algorithm',lambda: generate_genetic(train_comp, n_gen)),
    ]

    all_results = []
    all_cands   = {}
    for name, gen_fn in methods:
        print(f"\n  [{name}]")
        cands = gen_fn()
        res   = evaluate_generation_quality(cands, train_comp, name)
        all_results.append(res)
        all_cands[name] = cands

        # 保存CSV：原子分数 + wt% + 合金名
        cands_df  = pd.DataFrame(cands, columns=ELEMENTS)
        wt_matrix = np.array([atomic_to_wt(c) for c in cands]) * 100
        wt_df = pd.DataFrame(wt_matrix, columns=[f'{e}_wt%' for e in ELEMENTS])
        def _name(row):
            parts = [f"{e}{row[f'{e}_wt%']:.2f}"
                     for e in ELEMENTS if e!='Mg' and row[f'{e}_wt%']>=0.05]
            return f"Mg-{''.join(parts)}" if parts else "Mg"
        alloy_names = pd.concat([cands_df, wt_df], axis=1).apply(_name, axis=1)
        pd.concat([pd.DataFrame({'alloy_name': alloy_names}), cands_df, wt_df],
                  axis=1).to_csv(
            os.path.join(out_dir, f'candidates_{name.replace(" ","_").replace("+","")}.csv'),
            index=False)

    # ── 汇总表：MMD用Random search归一化 ─────────────────────
    random_mmd = next((r['mmd'] for r in all_results if 'Random' in r['label']), 1.0)
    for r in all_results:
        r['mmd_norm'] = r['mmd'] / (random_mmd + 1e-10)

    summary_df = pd.DataFrame([{k: v for k, v in r.items()
                                 if k not in ('cands', 'raw_diversity', 'mmd')}
                                for r in all_results])
    summary_df.to_csv(os.path.join(out_dir, 'generation_quality_summary.csv'), index=False)

    print(f"\n{'='*60}\nSUMMARY")
    print(f"  diversity/novelty>1.0=比训练集更好；MMD_norm<1.0=分布比Random更接近训练集")
    print(summary_df[['label', 'diversity', 'novelty',
                       'coverage', 'mmd_norm']].to_string(index=False))

    # ── 可视化 ────────────────────────────────────────────────
    plot_generation_quality(all_results, all_cands, train_comp, out_dir)
    return all_results, summary_df



# ══════════════════════════════════════════════════════════════
# 10. 可视化（NC风格，双栏宽18.3cm，Arial）
# ══════════════════════════════════════════════════════════════
def plot_generation_quality(results, all_cands, train_comp, out_dir):
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # ── NC-style rcParams ────────────────────────────────────────
    RC = {
        'font.family':        ['Arial', 'DejaVu Sans', 'sans-serif'],
        'font.size':          7,
        'axes.linewidth':     0.8,
        'axes.spines.top':    False,
        'axes.spines.right':  False,
        'xtick.major.width':  0.8,
        'ytick.major.width':  0.8,
        'xtick.major.size':   3.0,
        'ytick.major.size':   3.0,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'pdf.fonttype':       42,
        'svg.fonttype':       'none',
    }
    matplotlib.rcParams.update(RC)

    # ── 方法颜色与标签（elem_freq图复用）───────────────────────
    COLORS  = ['#1D9E75', '#4DBBD5', '#E64B35', '#3C5488', '#F39B7F']
    xlabels = ['WAE+MCMC\n(ours)', 'Random\nsearch',
               'Grid\nsearch', 'Bayesian\nOpt+EI', 'Genetic\nAlgorithm']

    # ── lollipop专用颜色 ─────────────────────────────────────
    TEAL     = '#1D9E75'   # WAE+MCMC
    GRAY_DOT = '#AAAAAA'   # baselines
    GRAY_STM = '#BBBBBB'   # baseline stems
    AX_COL   = '#1A1A1A'   # axis/tick color

    method_labels = ['WAE+MCMC\n(ours)', 'Random\nsearch',
                     'Grid\nsearch', 'Bayesian\nOpt+EI', 'Genetic\nAlgorithm']
    n_methods = len(results)

    # ══════════════════════════════════════════════════════════════
    # 图1：3指标 lollipop（NC单栏宽8.7cm，高5.5cm）
    # 去掉MMD_norm（语义暧昧），保留diversity / novelty / coverage
    # NC风格：L形轴（bottom+left），无grid，短tick，Arial字体
    # ══════════════════════════════════════════════════════════════
    metric_cfg = [
        # (key,       x_label,     panel, xmin,  xmax,  nticks)
        ('diversity', 'Diversity',  'a',   0.30,  1.05,  3),
        ('novelty',   'Novelty',    'b',   1.50,  5.50,  4),
        ('coverage',  'Coverage',   'c',   0.30,  1.02,  3),
    ]

    # 单栏宽 8.7 cm，高 5.5 cm；3个panel横排
    fig_w = 8.7 / 2.54
    fig_h = 5.5 / 2.54
    fig, axes = plt.subplots(
        1, 3, figsize=(fig_w, fig_h),
        gridspec_kw={'wspace': 0.55, 'left': 0.04, 'right': 0.96,
                     'top': 0.88, 'bottom': 0.14},
    )

    for ax, (key, xlabel, panel, xmin, xmax, nticks) in zip(axes, metric_cfg):
        vals = [r[key] for r in results]

        # ── 每个方法一行，从上到下排列 ──────────────────────────
        for i, v in enumerate(vals):
            is_ours = (i == 0)
            y       = n_methods - 1 - i          # 反转：WAE+MCMC在顶部
            color   = TEAL    if is_ours else GRAY_DOT
            stemcol = TEAL    if is_ours else GRAY_STM
            lw      = 2.0     if is_ours else 1.0
            ms      = 5.5     if is_ours else 3.8
            mew     = 0.8     if is_ours else 0.0
            mec     = 'white' if is_ours else 'none'

            # stem from xmin to value
            ax.plot([xmin, v], [y, y],
                    color=stemcol, lw=lw, solid_capstyle='round', zorder=2)
            # dot
            ax.plot(v, y,
                    'o', color=color, ms=ms,
                    mec=mec, mew=mew, zorder=3)
            # value label to the right
            ax.text(xmax + (xmax - xmin) * 0.03, y,
                    f'{v:.2f}',
                    va='center', ha='left',
                    fontsize=6.5 if is_ours else 6.0,
                    color=TEAL if is_ours else '#888888',
                    fontweight='bold' if is_ours else 'normal')

        # ── axes styling (NC: L-shape, no grid) ─────────────────
        ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.01)
        ax.set_ylim(-0.6, n_methods - 0.4)

        # x轴刻度
        xticks = np.linspace(xmin, xmax, nticks + 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'{v:.1f}' for v in xticks],
                           fontsize=6.5, color=AX_COL)

        # y轴：不显示数字，无刻度线（行标签靠legend说明）
        ax.set_yticks([])
        ax.yaxis.set_visible(False)

        # NC L形轴：保留bottom + left，去掉top/right
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_color(AX_COL)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['left'].set_color(AX_COL)
        ax.spines['left'].set_bounds(-0.5, n_methods - 0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.tick_params(axis='x', which='major',
                       length=3, width=0.8, color=AX_COL,
                       labelcolor=AX_COL, pad=2)

        # x轴标签（指标名）
        ax.set_xlabel(xlabel, fontsize=7.5, fontweight='bold',
                      labelpad=4, color=AX_COL)

        # panel letter
        ax.text(-0.12, 1.10, panel, transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', color=AX_COL)

    # ── 统一图例（放在图下方） ───────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=TEAL, markersize=5.5,
               markeredgecolor='white', markeredgewidth=0.8,
               label='WAE+MCMC (ours)'),
        Line2D([0], [0], marker='o', color='none',
               markerfacecolor=GRAY_DOT, markersize=3.8,
               label='Baselines'),
    ]
    fig.legend(handles=legend_elements,
               loc='lower center', ncol=2,
               fontsize=6.5, frameon=False,
               handletextpad=0.4, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.04))

    for ext in ('png', 'svg'):
        fig.savefig(os.path.join(out_dir, f'generation_metrics.{ext}'),
                    dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════
    # 图2：三元图小多图（Mg / RE / non-RE），5种方法各一个，2行3列布局
    # ══════════════════════════════════════════════════════════════
    import mpltern
    from matplotlib.lines import Line2D as _Line2D

    RE_SET     = {'Y', 'Nd', 'Ce', 'Gd', 'Sc', 'La'}
    _mg_idx    = ELEMENTS.index('Mg')
    _fe_idx    = ELEMENTS.index('Fe')
    _re_idx    = [i for i, e in enumerate(ELEMENTS) if e in RE_SET]
    _nonre_idx = [i for i, e in enumerate(ELEMENTS)
                  if i != _mg_idx and i != _fe_idx and e not in RE_SET]

    def _to_mg_re_nonre(comp_array):
        out = []
        for comp in comp_array:
            wt    = atomic_to_wt(comp)
            mg    = wt[_mg_idx]
            re    = sum(wt[i] for i in _re_idx)
            nonre = sum(wt[i] for i in _nonre_idx)
            s     = mg + re + nonre
            out.append([mg/s, re/s, nonre/s])
        return np.array(out)

    METHOD_STYLE_T = {
        'wae':    dict(color='#1D9E75', label='WAE+MCMC (ours)',   alpha=0.75, ms=8),
        'random': dict(color='#4DBBD5', label='Random search',     alpha=0.55, ms=6),
        'bayes':  dict(color='#3C5488', label='Bayesian Opt+EI',   alpha=0.55, ms=6),
        'grid':   dict(color='#E64B35', label='Grid search',       alpha=0.55, ms=6),
        'ga':     dict(color='#F39B7F', label='Genetic Algorithm', alpha=0.55, ms=6),
    }
    method_keys  = ['wae', 'random', 'grid', 'bayes', 'ga']
    method_comps = [results[i]['cands'] for i in range(len(results))]

    # 训练集成分
    _train_pts = _to_mg_re_nonre(train_comp)

    # 布局：2行3列，第2行只用前2个位置
    _nrows, _ncols = 2, 3
    _fig_w = 18.3 / 2.54
    _fig_h = _fig_w * 0.75
    fig2 = plt.figure(figsize=(_fig_w, _fig_h))

    _positions = [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1),
    ]

    for idx, mkey in enumerate(method_keys):
        ri, ci = _positions[idx]
        st = METHOD_STYLE_T[mkey]

        # 计算subplot位置（手动，给三角图留边距）
        _pad_l, _pad_r = 0.08, 0.04
        _pad_t, _pad_b = 0.06, 0.08
        _cell_w = (1.0 - _pad_l - _pad_r) / _ncols
        _cell_h = (1.0 - _pad_t - _pad_b) / _nrows
        _inner  = 0.08  # 每个格子内边距

        _left   = _pad_l + ci * _cell_w + _inner * _cell_w
        _bottom = 1.0 - _pad_t - (ri + 1) * _cell_h + _inner * _cell_h
        _width  = _cell_w * (1 - 2 * _inner)
        _height = _cell_h * (1 - 2 * _inner)

        ax2 = fig2.add_axes([_left, _bottom, _width, _height],
                            projection='ternary')

        # 训练集（灰色，底层）
        ax2.scatter(_train_pts[:,0], _train_pts[:,1], _train_pts[:,2],
                    marker='x', color='#CCCCCC', s=6,
                    linewidths=0.5, alpha=0.5, zorder=2)

        # 当前方法
        pts = _to_mg_re_nonre(method_comps[idx])
        ax2.scatter(pts[:,0], pts[:,1], pts[:,2],
                    marker='x', color=st['color'],
                    s=st['ms'], linewidths=0.7,
                    alpha=st['alpha'], zorder=5)

        # 坐标轴范围
        ax2.set_tlim(0.85, 1.0)
        ax2.set_llim(0.0,  0.15)
        ax2.set_rlim(0.0,  0.15)

        # 标签放在边上而非顶点
        ax2.taxis.label.set_text('Mg (wt.%)')
        ax2.taxis.label.set_fontsize(6)
        ax2.taxis.set_label_position('tick1')
        ax2.laxis.label.set_text('Rare earth elements (wt.%)')
        ax2.laxis.label.set_fontsize(6)
        ax2.laxis.set_label_position('tick1')
        ax2.raxis.label.set_text('Non-rare earth elements (wt.%)')
        ax2.raxis.label.set_fontsize(6)
        ax2.raxis.set_label_position('tick1')

        # tick：只保留一侧，去掉重复
        ax2.taxis.set_tick_params(labelsize=5, direction='in', length=2,
                                  tick1On=True, tick2On=False,
                                  label1On=True, label2On=False)
        ax2.laxis.set_tick_params(labelsize=5, direction='in', length=2,
                                  tick1On=True, tick2On=False,
                                  label1On=True, label2On=False)
        ax2.raxis.set_tick_params(labelsize=5, direction='in', length=2,
                                  tick1On=True, tick2On=False,
                                  label1On=False, label2On=False)

        # 网格
        ax2.grid(color='#E8E8E8', linewidth=0.3, linestyle='--')

        # 子图标题（方法名）
        ax2.set_title(st['label'], fontsize=6.5, color=st['color'],
                      fontweight='bold' if mkey == 'wae' else 'normal',
                      pad=4)

    # 第2行第3列空白位置加图例
    _left_leg  = _pad_l + 2 * _cell_w + _inner * _cell_w
    _bot_leg   = 1.0 - _pad_t - 2 * _cell_h + _inner * _cell_h
    _ax_leg    = fig2.add_axes([_left_leg, _bot_leg, _cell_w * 0.8, _cell_h * 0.8])
    _ax_leg.axis('off')

    _legend_handles = [
        _Line2D([0],[0], marker='x', color='#CCCCCC', linestyle='none',
                markersize=4, markeredgewidth=0.6, label='Training set')
    ]
    for mkey in method_keys:
        st = METHOD_STYLE_T[mkey]
        _legend_handles.append(
            _Line2D([0],[0], marker='x', color=st['color'], linestyle='none',
                    markersize=4, markeredgewidth=0.6, label=st['label'])
        )
    _ax_leg.legend(handles=_legend_handles,
                   loc='center', fontsize=6,
                   frameon=False, handlelength=1.0,
                   handletextpad=0.5, labelspacing=0.5)

    for ext in ('png', 'svg'):
        fig2.savefig(os.path.join(out_dir, f'generation_ternary.{ext}'),
                     dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)

    # ══════════════════════════════════════════════════════════
    # 图3：元素选用频率对比（双栏18.3cm）
    # 显示各元素被选为主合金元素的频率（wt%>0.3%）
    # 对比所有5种方法，能直观看出WAE更倾向于哪些元素
    # ══════════════════════════════════════════════════════════
    fe_idx = ELEMENTS.index('Fe')
    # 非Mg非Fe元素
    elem_indices = [i for i in range(1, len(ELEMENTS)) if i != fe_idx]
    elem_names   = [ELEMENTS[i] for i in elem_indices]

    # 计算各方法每个元素的选用频率（wt%>0.3%的比例）
    freq_train = []
    train_wt_full = np.array([atomic_to_wt(c) for c in train_comp]) * 100
    for i in elem_indices:
        freq_train.append(float(np.mean(train_wt_full[:, i] > 0.3)))

    freq_methods = []
    for res in results:
        wt_all = np.array([atomic_to_wt(c) for c in res['cands']]) * 100
        freq = [float(np.mean(wt_all[:, i] > 0.3)) for i in elem_indices]
        freq_methods.append(freq)

    # 只保留至少一个方法选用频率>2%的元素
    active_mask = np.array(freq_train) > 0.02
    for fm in freq_methods:
        active_mask |= np.array(fm) > 0.02
    active_elem_idx = [j for j, m in enumerate(active_mask) if m]
    active_names_f  = [elem_names[j] for j in active_elem_idx]

    n_elem = len(active_names_f)
    n_methods = len(results) + 1  # +1 for training set
    group_w = 0.8
    bar_w   = group_w / n_methods

    fig3, ax3 = plt.subplots(figsize=(18.3/2.54, 5.0/2.54),
                              gridspec_kw={'left': 0.07, 'right': 0.98,
                                           'top': 0.88, 'bottom': 0.18})
    pos = np.arange(n_elem)

    # 训练集（灰色）
    offsets = np.linspace(-group_w/2 + bar_w/2, group_w/2 - bar_w/2, n_methods)
    ax3.bar(pos + offsets[0],
            [freq_train[j] * 100 for j in active_elem_idx],
            width=bar_w * 0.9, color='#AAAAAA', label='Training set',
            linewidth=0, zorder=3)

    # 各方法
    for mi, (res, freq) in enumerate(zip(results, freq_methods)):
        lw = 0.8 if mi == 0 else 0.0
        ec = '#1A1A1A' if mi == 0 else 'none'
        ax3.bar(pos + offsets[mi + 1],
                [freq[j] * 100 for j in active_elem_idx],
                width=bar_w * 0.9, color=COLORS[mi],
                linewidth=lw, edgecolor=ec, zorder=3,
                label=xlabels[mi].replace('\n', ' '))

    ax3.set_xticks(pos)
    ax3.set_xticklabels(active_names_f, fontsize=7)
    ax3.set_ylabel('Selection frequency (%)', fontsize=7, labelpad=2)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(labelsize=6, pad=1)
    ax3.legend(fontsize=5.5, frameon=False, loc='upper right',
               ncol=3, handlelength=1, handletextpad=0.4,
               borderpad=0, labelspacing=0.3, columnspacing=0.8)
    ax3.text(-0.05, 1.10, '(f)', transform=ax3.transAxes,
             fontsize=8, fontweight='bold', va='top')

    for ext in ('png', 'svg'):
        fig3.savefig(os.path.join(out_dir, f'generation_elem_freq.{ext}'),
                     dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)

    print(f"\nFigures saved → {out_dir}/")
    print(f"  generation_metrics.(png/svg)   — lollipop metrics (a-c)")
    print(f"  generation_ternary.(png/svg)   — ternary composition space")
    print(f"  generation_elem_freq.(png/svg) — element selection frequency")


# ══════════════════════════════════════════════════════════════
# 11. 入口
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',  default=DATA_PATH)
    parser.add_argument('--wae',   default=WAE_WEIGHTS)
    parser.add_argument('--n_gen', type=int, default=500)
    parser.add_argument('--out',   default='generation_quality_results')
    args = parser.parse_args()

    DATA_PATH   = args.data
    WAE_WEIGHTS = args.wae
    N_GEN       = args.n_gen

    run_part1(n_gen=args.n_gen, out_dir=args.out)
    print("\nDone.")