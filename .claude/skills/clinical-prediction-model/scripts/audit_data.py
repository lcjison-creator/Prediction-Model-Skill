#!/usr/bin/env python3
"""临床预测模型 —— 第 1 步：数据质检。

先看清楚数据能撑起什么样的模型，再决定怎么建模。输出 audit.json（给流程下游读）
和 audit_report_zh.md（给研究者读）。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 缺失率超过这条线的变量，填补出来的值基本是编造的
MISSING_DROP = 0.40
MISSING_WARN = 0.20
# 两个变量相关性超过这条线，同时进模型会让系数不稳、重要性互相稀释
CORR_HIGH = 0.80
# 每个模型参数至少要有这么多事件，否则过拟合几乎必然发生
EPV_MIN = 10
# 事件数低于这条线，任何多变量模型都不可信
EVENTS_BLOCKING = 20


def load_table(path):
    p = Path(path)
    if not p.exists():
        sys.exit(f"找不到数据文件：{path}")
    if p.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030", "latin-1"):
        try:
            return pd.read_csv(p, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit(f"无法解码 {path}，请另存为 UTF-8 编码的 CSV")


def split_types(df, cols, forced_categorical):
    """区分连续变量和分类变量。

    数值型里 unique 值很少的（比如 0/1/2 的分级）容易被误当成连续变量，
    这里标记出来让研究者确认，而不是替他们做主。
    """
    numeric, categorical, ambiguous = [], [], []
    for c in cols:
        if c in forced_categorical:
            categorical.append(c)
            continue
        s = df[c]
        # pandas 3.0 起字符串列是 StringDtype 而不是 object，所以直接问"是不是数值"最稳
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            categorical.append(c)
            continue
        nun = s.nunique(dropna=True)
        vals = s.dropna()
        is_int_like = bool(len(vals)) and bool(np.all(np.equal(np.mod(vals.to_numpy(dtype=float), 1), 0)))
        if nun <= 2:
            categorical.append(c)
        elif nun <= 6 and is_int_like:
            categorical.append(c)
            ambiguous.append({"column": c, "n_unique": int(nun)})
        else:
            numeric.append(c)
    return numeric, categorical, ambiguous


def resolve_outcome(df, outcome, positive_label):
    s = df[outcome]
    vals = sorted(s.dropna().unique().tolist(), key=str)
    if len(vals) != 2:
        sys.exit(
            f"结局列 '{outcome}' 有 {len(vals)} 个取值：{vals[:10]}。"
            "本 skill 只处理二分类结局，请先把结局整理成两类。"
        )
    if positive_label is not None:
        pos = positive_label
        for v in vals:
            if str(v) == str(positive_label):
                pos = v
                break
        else:
            sys.exit(f"指定的阳性标签 '{positive_label}' 不在结局取值 {vals} 中")
    else:
        # 没指定就取"大的那个"：1 优于 0，'是' 优于 '否'（按排序取后者）
        pos = vals[1]
    y = (s == pos).astype(float)
    y[s.isna()] = np.nan
    return y, pos, vals


def count_params(df, numeric, categorical):
    """估算模型参数个数：连续变量各占 1 个，分类变量占 (类别数-1) 个。"""
    n = len(numeric)
    for c in categorical:
        n += max(1, df[c].nunique(dropna=True) - 1)
    return n


def high_corr_pairs(df, numeric):
    if len(numeric) < 2:
        return []
    corr = df[numeric].corr(method="spearman").abs()
    pairs = []
    for i, a in enumerate(numeric):
        for b in numeric[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= CORR_HIGH:
                pairs.append({"var1": a, "var2": b, "abs_spearman_r": round(float(r), 3)})
    return sorted(pairs, key=lambda d: -d["abs_spearman_r"])


def choose_strategy(n_rows, events, epv):
    """样本紧张时切测试集是浪费——小测试集的 AUC 置信区间宽到没有意义，
    不如全数据训练再用 bootstrap 估计乐观度（Harrell 内部验证法）。"""
    if events < EVENTS_BLOCKING:
        return "bootstrap", "事件数过少，任何切分都会让验证集失去意义；仅能做 bootstrap 内部验证，且结果需谨慎解读"
    if epv < EPV_MIN or n_rows < 500:
        return "bootstrap", f"EPV={epv:.1f}、样本量={n_rows}，切出测试集后剩余训练数据不足；采用全数据训练 + bootstrap 乐观度校正"
    if n_rows < 2000:
        return "split", f"样本量={n_rows} 且 EPV={epv:.1f} 充足，可切出 30% 独立验证集，并在训练集内再做 bootstrap 校正"
    return "split", f"样本量={n_rows} 充足，切出 25% 独立验证集，并在训练集内做 bootstrap 校正"


def main():
    ap = argparse.ArgumentParser(description="临床预测模型数据质检")
    ap.add_argument("--data", required=True, help="CSV 或 Excel 数据文件")
    ap.add_argument("--outcome", required=True, help="结局变量列名")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--exclude", default="", help="要排除的列，逗号分隔（如住院号、姓名）")
    ap.add_argument("--categorical", default="", help="强制指定为分类变量的列，逗号分隔")
    ap.add_argument("--positive-label", default=None, help="结局的阳性标签，默认取排序靠后的那个")
    args = ap.parse_args()

    out = Path(args.out)
    (out).mkdir(parents=True, exist_ok=True)

    df = load_table(args.data)
    if args.outcome not in df.columns:
        sys.exit(f"数据里没有列 '{args.outcome}'。现有列：{list(df.columns)}")

    excluded = [c.strip() for c in args.exclude.split(",") if c.strip()]
    forced_cat = [c.strip() for c in args.categorical.split(",") if c.strip()]
    missing_excluded = [c for c in excluded + forced_cat if c not in df.columns]
    if missing_excluded:
        sys.exit(f"这些列在数据中不存在：{missing_excluded}")

    y, pos_label, outcome_values = resolve_outcome(df, args.outcome, args.positive_label)
    n_missing_outcome = int(y.isna().sum())
    keep = y.notna()
    df, y = df[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    predictors = [c for c in df.columns if c != args.outcome and c not in excluded]

    # 常量列和近常量列进模型只会增加参数不提供信息
    constant, near_constant, id_like = [], [], []
    n_rows = len(df)
    for c in predictors:
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            constant.append(c)
        elif df[c].value_counts(normalize=True, dropna=True).iloc[0] >= 0.99:
            near_constant.append(c)
        elif nun == n_rows and df[c].dtype == object:
            id_like.append(c)

    auto_drop = set(constant + near_constant + id_like)
    predictors = [c for c in predictors if c not in auto_drop]

    numeric, categorical, ambiguous = split_types(df, predictors, forced_cat)

    missing = []
    for c in predictors:
        rate = float(df[c].isna().mean())
        if rate > 0:
            missing.append({"column": c, "missing_rate": round(rate, 4),
                            "n_missing": int(df[c].isna().sum())})
    missing.sort(key=lambda d: -d["missing_rate"])
    drop_for_missing = [m["column"] for m in missing if m["missing_rate"] > MISSING_DROP]
    warn_for_missing = [m["column"] for m in missing
                        if MISSING_WARN < m["missing_rate"] <= MISSING_DROP]

    usable = [c for c in predictors if c not in drop_for_missing]
    usable_num = [c for c in numeric if c in usable]
    usable_cat = [c for c in categorical if c in usable]

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    events = min(n_pos, n_neg)          # 少数类才是限制因素
    prevalence = n_pos / len(y) if len(y) else 0.0
    n_params = count_params(df, usable_num, usable_cat)
    epv = events / n_params if n_params else 0.0
    max_params = int(events / EPV_MIN)

    strategy, strategy_reason = choose_strategy(len(df), events, epv)
    corr_pairs = high_corr_pairs(df, usable_num)

    findings = []

    def add(level, code, msg, action):
        findings.append({"level": level, "code": code, "message": msg, "action": action})

    if events < EVENTS_BLOCKING:
        add("blocking", "too_few_events",
            f"少数类只有 {events} 例（阳性 {n_pos}、阴性 {n_neg}），低于 {EVENTS_BLOCKING} 例的下限。",
            "多变量模型在这个事件数下必然过拟合，得到的 AUC 无法复现。建议扩大队列、放宽结局定义使事件数上升，或改做描述性/单因素分析。")
    elif epv < EPV_MIN:
        add("critical", "low_epv",
            f"EPV = {epv:.1f}（{events} 个事件 / {n_params} 个模型参数），低于常用下限 {EPV_MIN}。",
            f"把候选变量对应的参数压到 {max_params} 个以内，优先按临床先验知识保留，不要用看着结局做的单因素筛选。")

    if prevalence < 0.10 or prevalence > 0.90:
        add("warning", "class_imbalance",
            f"结局患病率 {prevalence:.1%}，类别不平衡。",
            "用 class_weight='balanced' 或调整决策阈值处理。不要用 SMOTE——它会系统性抬高预测概率，让校准曲线失真，而校准是临床模型的核心指标。")

    if drop_for_missing:
        add("critical", "high_missing",
            f"这些变量缺失率超过 {MISSING_DROP:.0%}，已自动剔除：{drop_for_missing}",
            "填补出来的值基本是编造的。如果其中有临床上非做不可的变量，考虑改成'是否检测'这样的二分类变量入模。")
    if warn_for_missing:
        add("warning", "moderate_missing",
            f"这些变量缺失率在 {MISSING_WARN:.0%}~{MISSING_DROP:.0%} 之间：{warn_for_missing}",
            "保留但需在论文里说明填补方法。填补必须放在 Pipeline 内、按折拟合，否则是数据泄露。")
    if constant or near_constant:
        add("warning", "constant_columns",
            f"常量或近常量列（已自动剔除）：{constant + near_constant}",
            "这些列不提供信息，只会占用模型参数。")
    if id_like:
        add("warning", "id_like_columns",
            f"疑似标识列（每行取值都不同，已自动剔除）：{id_like}",
            "确认是否为住院号/姓名一类的标识符。")
    if corr_pairs:
        top = corr_pairs[:5]
        add("warning", "collinearity",
            f"存在高度共线的变量对（|Spearman r| ≥ {CORR_HIGH}）：" +
            "、".join(f"{p['var1']}~{p['var2']}(r={p['abs_spearman_r']})" for p in top),
            "每对里保留临床上更易获得、更常规检测的那一个。共线变量同时入模会让回归系数不稳、树模型的重要性互相稀释。")
    if ambiguous:
        add("info", "ambiguous_type",
            "这些数值列取值很少，可能其实是分级/分类变量：" +
            "、".join(f"{a['column']}({a['n_unique']}个取值)" for a in ambiguous),
            "已按分类变量处理。如果它们是有序等级且想按连续处理，用 --categorical 之外的方式确认。")
    if n_missing_outcome:
        add("warning", "missing_outcome",
            f"有 {n_missing_outcome} 例结局缺失，已排除。",
            "在论文的流程图（flow diagram）里要交代这部分病例的去向。")

    audit = {
        "data_file": str(args.data),
        "outcome": args.outcome,
        "outcome_values": [str(v) for v in outcome_values],
        "positive_label": str(pos_label),
        "n_rows": int(len(df)),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_events_minority": events,
        "prevalence": round(prevalence, 4),
        "n_missing_outcome_excluded": n_missing_outcome,
        "predictors_numeric": usable_num,
        "predictors_categorical": usable_cat,
        "n_predictors": len(usable_num) + len(usable_cat),
        "n_model_params": n_params,
        "epv": round(epv, 2),
        "max_params_at_epv10": max_params,
        "dropped_high_missing": drop_for_missing,
        "dropped_constant": constant + near_constant,
        "dropped_id_like": id_like,
        "excluded_by_user": excluded,
        "missing_by_column": missing,
        "high_correlation_pairs": corr_pairs,
        "recommended_strategy": strategy,
        "strategy_reason": strategy_reason,
        "findings": findings,
        "blocking": any(f["level"] == "blocking" for f in findings),
    }
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 人读版报告 ----
    icon = {"blocking": "🛑 严重", "critical": "❗ 重要", "warning": "⚠️ 提醒", "info": "ℹ️ 说明"}
    L = []
    L.append("# 数据质检报告\n")
    L.append(f"数据文件：`{args.data}`　结局变量：`{args.outcome}`（阳性 = `{pos_label}`）\n")
    L.append("## 一、样本概况\n")
    L.append("| 项目 | 数值 |")
    L.append("|---|---|")
    L.append(f"| 纳入例数 | {len(df)} |")
    L.append(f"| 阳性事件 | {n_pos} |")
    L.append(f"| 阴性 | {n_neg} |")
    L.append(f"| 患病率 | {prevalence:.1%} |")
    L.append(f"| 候选预测变量 | {len(usable_num) + len(usable_cat)} 个（连续 {len(usable_num)}、分类 {len(usable_cat)}）|")
    L.append(f"| 模型参数个数 | {n_params} |")
    L.append(f"| **EPV（每参数事件数）** | **{epv:.1f}** |")
    L.append(f"| EPV≥10 时最多可容纳参数 | {max_params} 个 |")
    L.append("")
    L.append("> EPV 是判断样本量够不够的核心指标：每个模型参数至少要有 10 个事件来支撑，"
             "否则模型会记住噪声而不是规律，在新病人身上直接失效。\n")
    L.append("## 二、推荐的验证策略\n")
    L.append(f"**{ {'bootstrap': '全数据训练 + bootstrap 乐观度校正', 'split': '切分独立验证集 + bootstrap 校正'}[strategy] }**\n")
    L.append(f"理由：{strategy_reason}\n")
    L.append("## 三、发现的问题\n")
    if findings:
        for f in findings:
            L.append(f"### {icon[f['level']]}：{f['message']}\n")
            L.append(f"**怎么办：** {f['action']}\n")
    else:
        L.append("没有发现明显问题，数据可以进入建模步骤。\n")
    if missing:
        L.append("## 四、各变量缺失情况\n")
        L.append("| 变量 | 缺失例数 | 缺失率 |")
        L.append("|---|---|---|")
        for m in missing[:30]:
            L.append(f"| {m['column']} | {m['n_missing']} | {m['missing_rate']:.1%} |")
        if len(missing) > 30:
            L.append(f"\n（仅显示缺失最多的 30 个，共 {len(missing)} 个变量存在缺失）")
        L.append("")
    L.append("## 五、进入建模的变量清单\n")
    L.append(f"**连续变量（{len(usable_num)}）：** {'、'.join(usable_num) if usable_num else '无'}\n")
    L.append(f"**分类变量（{len(usable_cat)}）：** {'、'.join(usable_cat) if usable_cat else '无'}\n")
    (out / "audit_report_zh.md").write_text("\n".join(L), encoding="utf-8")

    print(f"质检完成 → {out}/audit_report_zh.md")
    print(f"  例数 {len(df)}｜事件 {events}｜EPV {epv:.1f}｜推荐策略 {strategy}")
    for f in findings:
        if f["level"] in ("blocking", "critical"):
            print(f"  [{f['level']}] {f['message']}")
    if audit["blocking"]:
        print("\n数据存在 blocking 级问题，不建议直接进入建模步骤。")


if __name__ == "__main__":
    main()
