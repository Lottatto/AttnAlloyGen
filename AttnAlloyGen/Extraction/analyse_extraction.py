"""
LLM提取一致性分析脚本 v2
- 用合金成分+腐蚀速率 匹配 Excel ground truth
- 计算字段级别 accuracy、缺失率、数值误差
- 输出可直接用于Response Letter的统计数字

用法：直接运行，修改下面两个路径即可
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ============================================================
BASE_DIR = r"D:/vscode/corrosion rate/revision/Extraction/results"
GT_PATH  = r"D:/vscode/corrosion rate/revision/Extraction/database_colab_text_new1.xlsx"
RUNS     = ["Kimi1", "Kimi2", "Kimi3"]
# ============================================================

# Excel里的文本字段（ground truth列名）
GT_TEXT_COLS = [
    "Test method",
    "Immersion Time",
    "Deformation Process",
    "Heat Treatment Process",
    "Heat Treatment Time",
    "Heat Treatment Temperature",
]

# JSON里对应的字段名（和上面一一对应）
JSON_TEXT_FIELDS = [
    "Test method",
    "Immersion Time",
    "Deformation process",
    "Heat Treatment Process",
    "Heat Treatment Time",
    "Heat Treatment Temperature",
]

JSON_COMP_FIELD = "Alloy Composition"
JSON_CR_FIELD   = "Corrosion Rate"

ELEMENT_COLS = ["Mg","Al","Zn","Li","Y","Mn","Nd","Ce","Ca","Gd","Zr","Sc","Fe","Sn","In","La"]
GT_CR_COL    = "Corrosion Rate (mm/y)"
GT_DOI_COL   = "DOI"


# ── 工具函数 ──────────────────────────────────────────────────

def normalize_text(v):
    if v is None: return ""
    s = str(v).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_cr(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    m = re.search(r"[\d.]+", str(v))
    return float(m.group()) if m else None

def parse_composition_str(s):
    """解析 'Mg-7Sn-1Ca-1Ag' -> {'Sn': 7.0, 'Ca': 1.0, 'Ag': 1.0}"""
    if not s: return {}
    comps = {}
    for m in re.finditer(r"([A-Z][a-z]?)([\d.]*)", str(s)):
        elem, num = m.group(1), m.group(2)
        if elem == "Mg": continue
        comps[elem] = float(num) if num else 0.0
    return comps

# 原子量表（用于at% -> wt%转换）
ATOMIC_MASS = {
    "Mg": 24.305, "Al": 26.982, "Zn": 65.38,  "Li": 6.941,
    "Y":  88.906, "Mn": 54.938, "Nd": 144.242, "Ce": 140.116,
    "Ca": 40.078, "Gd": 157.25, "Zr": 91.224,  "Sc": 44.956,
    "Fe": 55.845, "Sn": 118.71, "In": 114.818,  "La": 138.905,
    "Ag": 107.868,
}

def at_fraction_to_wt_pct(at_fractions, elem_cols):
    """
    把Excel行的at fraction转成wt%
    at_fractions: dict {elem: at_fraction}
    返回: dict {elem: wt%}
    """
    # 计算各元素质量贡献
    mass_contribs = {}
    for elem in elem_cols:
        at_f = float(at_fractions.get(elem, 0))
        mass = ATOMIC_MASS.get(elem, 1.0)
        mass_contribs[elem] = at_f * mass
    total_mass = sum(mass_contribs.values())
    if total_mass == 0:
        return {e: 0.0 for e in elem_cols}
    return {elem: (m / total_mass) * 100 for elem, m in mass_contribs.items()}

def comp_distance(comp_dict_wt, gt_row):
    """
    comp_dict_wt: {elem: wt%}，来自JSON字符串解析
    gt_row: Excel行，元素列是at fraction，需先转wt%
    """
    gt_wt = at_fraction_to_wt_pct(
        {e: gt_row[e] for e in ELEMENT_COLS if e in gt_row.index},
        ELEMENT_COLS
    )
    dist = 0.0
    for elem, val in comp_dict_wt.items():
        gt_val = gt_wt.get(elem, 0.0)
        dist += (val - gt_val) ** 2
    return dist ** 0.5

# 目录文件名索引缓存，避免重复glob
_dir_index_cache = {}

def build_dir_index(run_dir):
    """建立 stem->Path 索引，stem是去掉.pdf.txt或.txt后的文件名"""
    key = str(run_dir)
    if key in _dir_index_cache:
        return _dir_index_cache[key]
    index = {}
    for f in run_dir.glob("*.txt"):
        fname = f.name
        if fname.endswith(".pdf.txt"):
            stem = fname[:-8]
        elif fname.endswith(".txt"):
            stem = fname[:-4]
        else:
            stem = fname
        index[stem] = f
    _dir_index_cache[key] = index
    return index

def find_json_file(run_dir, doi):
    """用目录索引查找对应.txt文件"""
    doi_clean = doi.strip().replace(" ", "")
    prefix = doi_clean.split("/")[0]
    suffix = doi_clean.split("/")[-1]
    prefix_nodot = prefix.replace(".", "")

    stem_candidates = [
        doi_clean.replace("/", "."),
        doi_clean.replace("/", ""),
        doi_clean.replace("/", "_"),
        doi_clean.replace("/", "-"),
        prefix_nodot + suffix,
        prefix + suffix,
        doi_clean,
    ]

    index = build_dir_index(run_dir)
    for stem in stem_candidates:
        if stem in index:
            return index[stem]
    return None

def load_json_records(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    # 找到第一个 [ 或 { 开始解析，跳过前面的说明文字
    start = -1
    for i, c in enumerate(raw):
        if c in "[{":
            start = i
            break
    if start == -1:
        return []
    try:
        content = json.loads(raw[start:])
    except json.JSONDecodeError:
        # 尝试提取最后一个完整JSON块
        import re
        matches = list(re.finditer(r'\[', raw))
        for m in reversed(matches):
            try:
                content = json.loads(raw[m.start():])
                break
            except:
                continue
        else:
            return []
    if isinstance(content, list):
        return content
    for key in ["records","data","entries","results","alloys"]:
        if isinstance(content, dict) and key in content:
            return content[key]
    return [content] if isinstance(content, dict) else []

def text_similarity(a, b):
    """字符级相似度，用于模糊匹配热处理字段"""
    a, b = normalize_text(a), normalize_text(b)
    if a == b: return 1.0
    if not a or not b: return 0.0
    common = sum(1 for c in a if c in b)
    return common / max(len(a), len(b))

def match_record_to_gt(record, gt_subset):
    """
    主键：合金成分（wt%距离）+ 热处理工艺（模糊匹配）
    腐蚀速率只作辅助（当成分+热处理打平时用）
    """
    comp_dict = parse_composition_str(record.get(JSON_COMP_FIELD, ""))
    ht_json   = normalize_text(record.get("Heat Treatment Process", ""))
    cr_json   = parse_cr(record.get(JSON_CR_FIELD))

    best_idx, best_score = None, float("inf")

    for idx, row in gt_subset.iterrows():
        # 1. 成分距离（wt%空间）
        comp_dist = comp_distance(comp_dict, row) if comp_dict else 0.0

        # 2. 热处理相似度 -> 距离
        ht_gt   = normalize_text(row.get("Heat Treatment Process", ""))
        ht_dist = 1.0 - text_similarity(ht_json, ht_gt)

        # 3. 腐蚀速率辅助
        cr_gt = parse_cr(row[GT_CR_COL])
        if cr_json and cr_gt and cr_gt > 0:
            cr_err = abs(cr_json - cr_gt) / cr_gt
        else:
            cr_err = 0.5

        # 综合得分：成分权重最高，热处理次之，腐蚀速率辅助
        score = comp_dist * 5.0 + ht_dist * 3.0 + cr_err * 1.0

        if score < best_score:
            best_score = score
            best_idx = idx

    # 成分距离 < 3 wt% 才认为匹配成功
    if best_idx is not None:
        comp_dist_best = comp_distance(comp_dict, gt_subset.loc[best_idx]) if comp_dict else 0.0
        if comp_dist_best < 3.0:
            return best_idx
    return None


# ── 主函数 ───────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("LLM提取一致性 & 准确率分析")
    print("=" * 65)

    gt = pd.read_excel(GT_PATH)
    gt[GT_DOI_COL] = gt[GT_DOI_COL].str.strip()
    print(f"GT: {len(gt)} 条记录，{gt[GT_DOI_COL].nunique()} 个DOI\n")

    # 先打印一个文件名样例，确认格式
    sample_doi = gt[GT_DOI_COL].iloc[0]
    sample_run_dir = Path(BASE_DIR) / RUNS[0]
    sample_file = find_json_file(sample_run_dir, sample_doi)
    print(f"样例DOI: {sample_doi}")
    print(f"找到文件: {sample_file}")
    if sample_run_dir.exists():
        all_files = list(sample_run_dir.glob("*.txt"))
        print(f"Kimi1目录下共 {len(all_files)} 个JSON文件")
        if all_files:
            print(f"文件名样例: {all_files[0].name}")
    print()

    field_results      = defaultdict(list)   # gf -> [(gt_val, json_val)]
    consistency_results = defaultdict(list)  # gf -> [bool]
    missing_stats      = defaultdict(lambda: [0, 0])  # gf -> [missing, total]
    matched_total = 0
    unmatched_total = 0
    doi_not_found = []

    for doi in gt[GT_DOI_COL].unique():
        gt_subset = gt[gt[GT_DOI_COL] == doi].copy()

        run_records = {}
        for run in RUNS:
            jf = find_json_file(Path(BASE_DIR) / run, doi)
            if jf:
                try:
                    run_records[run] = load_json_records(jf)
                except Exception as e:
                    print(f"  [错误] {run}/{jf.name}: {e}")
                    run_records[run] = []
            else:
                run_records[run] = []

        if not any(run_records.values()):
            doi_not_found.append(doi)
            continue

        primary_records = run_records.get(RUNS[0], [])

        for rec in primary_records:
            # 缺失统计（所有记录都算）
            for jf, gf in zip(JSON_TEXT_FIELDS, GT_TEXT_COLS):
                val = rec.get(jf, None)
                missing_stats[gf][1] += 1
                if val is None or str(val).strip() == "":
                    missing_stats[gf][0] += 1

            gt_idx = match_record_to_gt(rec, gt_subset)
            if gt_idx is None:
                unmatched_total += 1
                continue

            matched_total += 1
            gt_row = gt.loc[gt_idx]

            # 准确率
            for jf, gf in zip(JSON_TEXT_FIELDS, GT_TEXT_COLS):
                json_val = normalize_text(rec.get(jf, ""))
                gt_val   = normalize_text(gt_row[gf])
                field_results[gf].append((gt_val, json_val))

            # 三次一致性：在Kimi2/3里找腐蚀速率最近的记录
            other_recs = {}
            for run in RUNS[1:]:
                cr1 = parse_cr(rec.get(JSON_CR_FIELD))
                best, best_s = None, float("inf")
                for r2 in run_records.get(run, []):
                    cr2 = parse_cr(r2.get(JSON_CR_FIELD))
                    if cr1 and cr2:
                        s = abs(cr1 - cr2) / (cr1 + 1e-9)
                        if s < best_s:
                            best_s = s
                            best = r2
                if best and best_s < 0.20:
                    other_recs[run] = best

            if len(other_recs) == len(RUNS) - 1:
                all_recs = [rec] + [other_recs[r] for r in RUNS[1:]]
                for jf, gf in zip(JSON_TEXT_FIELDS, GT_TEXT_COLS):
                    vals = [normalize_text(r.get(jf, "")) for r in all_recs]
                    consistency_results[gf].append(len(set(vals)) == 1)

    # ── 输出 ─────────────────────────────────────────────────

    print(f"匹配到GT的记录：{matched_total}  未匹配：{unmatched_total}\n")
    print("=" * 65)
    print(f"{'字段':<32} {'准确率':>8} {'缺失率':>8} {'三次一致率':>12}")
    print("-" * 65)

    summary = {}
    for gf in GT_TEXT_COLS:
        pairs = field_results[gf]
        acc = sum(1 for g,j in pairs if g==j) / len(pairs) if pairs else None
        miss, total = missing_stats[gf]
        miss_rate = miss / total if total > 0 else None
        cons = consistency_results[gf]
        cons_rate = sum(cons) / len(cons) if cons else None
        summary[gf] = dict(accuracy=acc, missing=miss_rate, consistency=cons_rate,
                           n=len(pairs))
        print(f"{gf:<32} "
              f"{acc:.1%} ({len(pairs)})" if acc is not None else f"{gf:<32} {'N/A':>8}",
              end="")
        print(f" {miss_rate:>8.1%}" if miss_rate is not None else f" {'N/A':>8}", end="")
        print(f" {cons_rate:>12.1%}" if cons_rate is not None else f" {'N/A':>12}")

    accs  = [v["accuracy"]    for v in summary.values() if v["accuracy"]    is not None]
    conss = [v["consistency"] for v in summary.values() if v["consistency"] is not None]
    if accs:
        print("-" * 65)
        print(f"{'整体平均':<32} {sum(accs)/len(accs):>8.1%} {'':>8} "
              f"{sum(conss)/len(conss):>12.1%}")

    # 典型错误案例
    print("\n" + "=" * 65)
    print("典型错误案例（每字段前3个）")
    print("=" * 65)
    for gf in GT_TEXT_COLS:
        errors = [(g,j) for g,j in field_results[gf] if g != j]
        if not errors: continue
        print(f"\n[{gf}]  共{len(errors)}个错误 / {len(field_results[gf])}个对比")
        for g, j in errors[:3]:
            print(f"  GT:   '{g}'")
            print(f"  JSON: '{j}'")

    # 找不到JSON文件的DOI
    if doi_not_found:
        print(f"\n⚠️  以下 {len(doi_not_found)} 个DOI在Kimi1/2/3里都没找到对应JSON：")
        index = build_dir_index(Path(BASE_DIR) / RUNS[0])
        actual_stems = sorted(index.keys())
        for d in doi_not_found[:10]:
            doi_clean = d.strip().replace(" ", "")
            prefix = doi_clean.split("/")[0]
            suffix = doi_clean.split("/")[-1]
            prefix_nodot = prefix.replace(".", "")
            candidates = [
                doi_clean.replace("/", "."),
                doi_clean.replace("/", ""),
                doi_clean.replace("/", "_"),
                prefix_nodot + suffix,
                prefix + suffix,
            ]
            print(f"  DOI: {d}")
            print(f"  尝试stem: {candidates[0]} | {candidates[1]}")
            close = [s for s in actual_stems if suffix[:10] in s]
            print(f"  目录中相似: {close[:2]}")
            # 精确验证：目标stem是否在index里
            for cand in candidates:
                if cand in index:
                    print(f"  ✓ index中找到: {cand} -> {index[cand].name}")
                    break
            else:
                print(f"  ✗ index中均未找到，index大小={len(index)}")
                # 打印index里的前3个key供对比
                sample_keys = [k for k in actual_stems if '108487' in k or 'jma.2013' in k or '110988' in k or 'corcom' in k]
                if sample_keys:
                    print(f"  index中相关key: {sample_keys[:3]}")
        if len(doi_not_found) > 10:
            print(f"   ...（共{len(doi_not_found)}个）")

    # 保存
    out = Path(BASE_DIR).parent / "extraction_analysis_results.txt"
    lines = ["LLM提取准确率分析", f"匹配: {matched_total}  未匹配: {unmatched_total}", "",
             f"{'字段':<32} {'准确率':>8} {'缺失率':>8} {'三次一致率':>12}", "-"*65]
    for gf, v in summary.items():
        a = f"{v['accuracy']:.1%} (n={v['n']})" if v["accuracy"] is not None else "N/A"
        m = f"{v['missing']:.1%}"  if v["missing"]     is not None else "N/A"
        c = f"{v['consistency']:.1%}" if v["consistency"] is not None else "N/A"
        lines.append(f"{gf:<32} {a:>14} {m:>8} {c:>12}")
    if accs:
        lines += ["", f"整体准确率均值: {sum(accs)/len(accs):.1%}",
                      f"整体一致率均值: {sum(conss)/len(conss):.1%}"]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 结果已保存: {out}")
    print("\n⚠️  如果'匹配到GT的记录'数量很少，把Kimi1里任一文件名发给我，我修正文件名规则。")


if __name__ == "__main__":
    main()