import os, re, sys, warnings, argparse
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import WenAlloys

# ── 路径 ──────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_PATH = os.path.join(SCRIPT_DIR, 'candidates_WAEMCMC.csv')
DATA_PATH       = os.path.join(SCRIPT_DIR, 'database_colab_text_new1.xlsx')
OUT_PATH        = os.path.join(SCRIPT_DIR, 'candidates_WAEMCMC_full.csv')

# ── 常量（与 al_ablation.py 完全一致）──────────────────────────
ELEMENTS = ['Mg','Al','Zn','Li','Y','Mn','Nd','Ce','Ca','Gd','Zr','Sc','Fe','Sn','In','La']
TARGET_COL = 'Corrosion Rate (mm/y)'
DOI_COL    = 'DOI'
CATEGORICAL_COLUMNS = [
    'Test method','Deformation Process','Heat Treatment Process',
    'Heat Treatment Time','Heat Treatment Temperature','Immersion Time'
]
DESCRIPTOR_COLS = [
    'Atomic weight mean', 'Electronegativity local mismatch', 'VEC mean',
    'Average Ionization Energy', 'Average Work Function',
    'Average IE Difference from Mg', 'Density', 'Average Melting Point',
    'Average Boiling Point', 'Average Fermi Energy', 'Average Atomic Radius',
    'Average Surface Energy', 'Average Adsorption Energy',
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

# ── 物理常数数组（与 al_ablation.py 完全一致）──────────────────
ATOMIC_WEIGHTS = {e: Element(e).atomic_mass.real for e in ELEMENTS}
_aw = np.array([ATOMIC_WEIGHTS[e] for e in ELEMENTS])

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

_ie_arr  = np.array([Element(e).ionization_energies[0] for e in ELEMENTS])
_wf_arr  = np.array([_WF.get(e, 0)  for e in ELEMENTS])
_mp_arr  = np.array([_MP.get(e, 0)  for e in ELEMENTS])
_bp_arr  = np.array([_BP.get(e, 0)  for e in ELEMENTS])
_fe_arr  = np.array([_FE.get(e, 0)  for e in ELEMENTS])
_se_arr  = np.array([_SE.get(e, 0)  for e in ELEMENTS])
_ads_arr = np.array([_ADS.get(e, 0) for e in ELEMENTS])
_ar_arr  = np.array([Element(e).atomic_radius.real for e in ELEMENTS])
_mg_ie   = Element('Mg').ionization_energies[0]
_dens_aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
_dens_d  = np.array([_DENS.get(e, 1) for e in ELEMENTS])

_wa = WenAlloys()


# ══════════════════════════════════════════════════════════════
# 1. 数值描述符计算
# ══════════════════════════════════════════════════════════════
def compute_descriptors(comp_array):
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
# 2. 数据加载（与 al_ablation.py 完全一致）
# ══════════════════════════════════════════════════════════════
def _std_immersion(t):
    t = str(t).strip().lower()
    for p, fn in [(r'(\d+(?:\.\d+)?)\s*h(?:ours?)?', lambda m: f"{float(m.group(1)):.0f} hours"),
                  (r'(\d+(?:\.\d+)?)\s*d(?:ays?)?',  lambda m: f"{int(float(m.group(1))*24)} hours")]:
        m = re.search(p, t)
        if m: return fn(m)
    return t

def _std_ht_proc(t):
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

def load_train_data(path=DATA_PATH):
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
    for col, fn in [('Immersion Time', _std_immersion),
                    ('Heat Treatment Process', _std_ht_proc),
                    ('Heat Treatment Temperature', _std_temp),
                    ('Heat Treatment Time', _std_time),
                    ('Deformation Process', _std_deform)]:
        if col in df.columns:
            df[col] = df[col].apply(fn)
    df = df[~df[DOI_COL].isin(EXCLUDED_DOIS)].copy()
    df['year'] = df[DOI_COL].map(DOI_YEAR_MAP)
    df = df[df['year'].notna()].reset_index(drop=True)
    # temporal split：只用训练集（year < 2025）做相似度匹配
    train_df = df[df['year'] < 2025].copy().reset_index(drop=True)
    return train_df


# ══════════════════════════════════════════════════════════════
# 3. 补全特征（build_candidate_df 逻辑）
# ══════════════════════════════════════════════════════════════
def build_full_candidates(cand_df, train_df):
    """
    输入: cand_df   — 候选池DataFrame（含ELEMENTS列，原子分数）
          train_df  — 训练集DataFrame（含ELEMENTS、CATEGORICAL_COLUMNS、DESCRIPTOR_COLS）
    输出: 补全了数值特征和文本特征的DataFrame
    """
    cand_comps = cand_df[ELEMENTS].values.astype(float)
    cand_comps = cand_comps / (cand_comps.sum(axis=1, keepdims=True) + 1e-10)

    train_comp = train_df[ELEMENTS].values.astype(float)
    train_comp_norm = train_comp / (train_comp.sum(axis=1, keepdims=True) + 1e-10)

    print(f"  计算 {len(cand_comps)} 个候选成分的数值描述符...")
    desc_df = compute_descriptors(cand_comps)

    print(f"  相似度匹配文本特征（训练集 {len(train_df)} 条）...")
    rows = []
    for i, cand in enumerate(cand_comps):
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(cand_comps)}")

        # 加权Euclidean距离（主合金元素权重3.0，与al_ablation.py一致）
        weights = np.ones(len(ELEMENTS))
        non_mg = cand[1:]
        main_i = int(np.argmax(non_mg)) + 1
        weights[main_i] = 3.0
        dists = np.sqrt(((train_comp_norm - cand) ** 2 * weights).sum(axis=1))
        ref   = train_df.iloc[np.argmin(dists)]

        row = {}
        # 成分（原子分数）
        for i_e, e in enumerate(ELEMENTS):
            row[e] = cand[i_e]
        # 数值描述符（重新计算）
        for col in DESCRIPTOR_COLS:
            row[col] = desc_df.iloc[i][col]
        # 文本特征（最近邻复制）
        for col in CATEGORICAL_COLUMNS:
            row[col] = ref[col]
        # 腐蚀速率占位（待预测/实验填充）
        row[TARGET_COL] = 0.0
        # 保留alloy_name和wt%列
        if 'alloy_name' in cand_df.columns:
            row['alloy_name'] = cand_df.iloc[i]['alloy_name']
        for e in ELEMENTS:
            wt_col = f'{e}_wt%'
            if wt_col in cand_df.columns:
                row[wt_col] = cand_df.iloc[i][wt_col]
        # 记录最近邻信息（方便debug）
        row['nn_dist']   = float(dists[np.argmin(dists)])
        row['nn_doi']    = ref.get(DOI_COL, '')

        rows.append(row)

    result_df = pd.DataFrame(rows)

    # 列顺序：alloy_name → 成分 → wt% → 数值描述符 → 文本特征 → 目标列
    col_order = (['alloy_name'] if 'alloy_name' in result_df.columns else [])
    col_order += ELEMENTS
    col_order += [f'{e}_wt%' for e in ELEMENTS if f'{e}_wt%' in result_df.columns]
    col_order += DESCRIPTOR_COLS
    col_order += CATEGORICAL_COLUMNS
    col_order += [TARGET_COL, 'nn_dist', 'nn_doi']
    col_order  = [c for c in col_order if c in result_df.columns]
    result_df  = result_df[col_order]

    return result_df


# ══════════════════════════════════════════════════════════════
# 4. 主流程
# ══════════════════════════════════════════════════════════════
ITER_FILES = [
    os.path.join(SCRIPT_DIR, '1st_iteration.xlsx'),
    os.path.join(SCRIPT_DIR, '2nd_iteration.xlsx'),
    os.path.join(SCRIPT_DIR, '3rd_iteration.xlsx'),
    os.path.join(SCRIPT_DIR, '4th_iteration.xlsx'),
]
ITER_NAMES = ['1st_iteration', '2nd_iteration', '3rd_iteration', '4th_iteration']


def main(candidates_path=CANDIDATES_PATH, data_path=DATA_PATH, out_path=OUT_PATH):
    print("=" * 60)
    print("build_candidates_full.py")
    print("=" * 60)

    print(f"\n[1] 加载候选池: {candidates_path}")
    cand_df = pd.read_csv(candidates_path)
    cand_df['source'] = 'candidates'
    print(f"    {len(cand_df)} 条候选成分")

    print(f"\n[2] 加载训练数据: {data_path}")
    train_df = load_train_data(data_path)
    print(f"    训练集（year<2025, pw）: {len(train_df)} 条")

    print(f"\n[3] 加载iteration实验成分...")
    iter_dfs = []
    for fpath, fname in zip(ITER_FILES, ITER_NAMES):
        if not os.path.exists(fpath):
            print(f"    警告：{fpath} 不存在，跳过")
            continue
        df_i = pd.read_excel(fpath)
        # 去掉腐蚀速率列（不需要真实值，只预测）
        df_i = df_i.drop(columns=[TARGET_COL], errors='ignore')
        df_i['source'] = fname
        # alloy_name如果没有就留空
        if 'alloy_name' not in df_i.columns:
            df_i['alloy_name'] = ''
        iter_dfs.append(df_i)
        print(f"    {fname}: {len(df_i)} 条")

    # 合并候选池和iteration
    all_dfs = [cand_df] + iter_dfs
    combined_df = pd.concat(all_dfs, ignore_index=True)
    for e in ELEMENTS:
        if e not in combined_df.columns:
            combined_df[e] = 0.0
    combined_df[ELEMENTS] = combined_df[ELEMENTS].fillna(0.0)
    print(f"\n    合并后总计: {len(combined_df)} 条")

    # 合并后统一计算wt%（at→wt转换，候选池和iteration都算）
    _aw = np.array([Element(e).atomic_mass.real for e in ELEMENTS])
    comps = combined_df[ELEMENTS].values.astype(float)
    comps_norm = comps / (comps.sum(axis=1, keepdims=True) + 1e-10)
    wt_matrix = comps_norm * _aw
    wt_matrix = wt_matrix / (wt_matrix.sum(axis=1, keepdims=True) + 1e-10) * 100
    for j, e in enumerate(ELEMENTS):
        combined_df[f'{e}_wt%'] = wt_matrix[:, j]

    print(f"\n[4] 补全特征...")
    full_df = build_full_candidates(combined_df, train_df)
    # 把source列加回来（build_full_candidates不处理source）
    full_df['source'] = combined_df['source'].values

    print(f"\n[5] 保存结果: {out_path}")
    full_df.to_csv(out_path, index=False)
    print(f"    输出列: {full_df.columns.tolist()}")
    print(f"    输出行数: {len(full_df)} "
          f"（候选池{(full_df['source']=='candidates').sum()}条 + "
          f"iteration {(full_df['source']!='candidates').sum()}条）")
    print("\n完成。")
    return full_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', default=CANDIDATES_PATH)
    parser.add_argument('--data',       default=DATA_PATH)
    parser.add_argument('--out',        default=OUT_PATH)
    args = parser.parse_args()
    main(args.candidates, args.data, args.out)