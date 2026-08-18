#!/usr/bin/env python3
"""临床预测模型 —— 第 2 步：建模、校准与内部验证。

设计上的两条硬规则：
1. 所有预处理（填补、编码、标准化）都在 sklearn Pipeline 内部，因此交叉验证和
   bootstrap 的每一次重抽样都只用当次训练数据拟合填补器。先在全数据上填补再切分
   是数据泄露，也是这类论文最常见的致命错误。
2. Logistic 回归永远作为基准模型一起跑。审稿人必问"机器学习比传统回归好在哪"，
   没有基准就没法回答。
"""
import argparse
import json
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from audit_data import load_table, resolve_outcome, split_types  # noqa: E402

warnings.filterwarnings("ignore")
EPS = 1e-6


# ---------------------------------------------------------------- 指标计算

def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """把 logit(预测概率) 当作唯一自变量重新拟合 logistic 回归。

    斜率 1、截距 0 是完美校准。斜率 < 1 说明预测值被拉得过于分散——高风险的估太高、
    低风险的估太低，是过拟合的典型表现。截距偏离 0 说明整体高估或低估了风险水平。
    """
    x = _logit(p).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    lr.fit(x, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def hosmer_lemeshow(y, p, g=10):
    """经典拟合优度检验。样本量大时几乎必然显著，所以只作参考，
    校准曲线和校准斜率才是主要依据。"""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    try:
        df["grp"] = pd.qcut(df["p"].rank(method="first"), g, labels=False)
    except ValueError:
        return float("nan"), float("nan")
    stat = 0.0
    for _, sub in df.groupby("grp"):
        obs, exp, n = sub["y"].sum(), sub["p"].sum(), len(sub)
        denom = exp * (1 - exp / n) if n and exp > 0 else 0
        if denom > 0:
            stat += (obs - exp) ** 2 / denom
    dof = max(g - 2, 1)
    return float(stat), float(1 - stats.chi2.cdf(stat, dof))


def threshold_metrics(y, p, thr):
    y = np.asarray(y, dtype=int)
    pred = (np.asarray(p) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    safe = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    return {
        "threshold": float(thr), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": safe(tp, tp + fn), "specificity": safe(tn, tn + fp),
        "PPV": safe(tp, tp + fp), "NPV": safe(tn, tn + fn),
        "accuracy": safe(tp + tn, len(y)),
        "F1": safe(2 * tp, 2 * tp + fp + fn),
    }


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def auc_ci_bootstrap(y, p, n_boot, rng):
    """对预测结果重抽样估计 AUC 的置信区间。不需要重新训练模型，因此可以多抽几次。"""
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def net_benefit_curve(y, p, thresholds):
    """决策曲线分析：把假阳性按 t/(1-t) 的权重折算后从真阳性里扣除，
    得到"净获益"。它回答的是"用这个模型做决策，比一刀切全治或全不治好多少"。"""
    y = np.asarray(y, dtype=int)
    n, prev = len(y), y.mean()
    rows = []
    for t in thresholds:
        pred = (np.asarray(p) >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        w = t / (1 - t) if t < 1 else np.inf
        rows.append({
            "threshold": float(t),
            "net_benefit_model": float(tp / n - fp / n * w),
            "net_benefit_all": float(prev - (1 - prev) * w),
            "net_benefit_none": 0.0,
        })
    return rows


def delong_test(y, p1, p2):
    """DeLong 检验：两个模型的 AUC 差异是否有统计学意义。
    用的是 Sun & Xu (2014) 的快速算法。"""
    y = np.asarray(y, dtype=int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan"), float("nan")

    def midrank(x):
        order = np.argsort(x)
        s = x[order]
        N = len(x)
        r = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N - 1 and s[j + 1] == s[i]:
                j += 1
            r[i:j + 1] = 0.5 * (i + j) + 1
            i = j + 1
        out = np.empty(N)
        out[order] = r
        return out

    preds = np.vstack([np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)])
    k = preds.shape[0]
    tx = np.array([midrank(preds[r, pos]) for r in range(k)])
    ty = np.array([midrank(preds[r, neg]) for r in range(k)])
    tz = np.array([midrank(preds[r, np.r_[pos, neg]]) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    s = np.atleast_2d(s)
    contrast = np.array([[1, -1]])
    var = float(np.atleast_2d(contrast @ s @ contrast.T)[0, 0])
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        return diff, float("nan")
    z = diff / np.sqrt(var)
    return diff, float(2 * (1 - stats.norm.cdf(abs(z))))


# ---------------------------------------------------------------- 模型构建

def build_preprocessor(numeric, categorical, scale):
    """填补策略：连续变量用中位数（对偏态和离群值稳健，临床检验指标普遍偏态），
    分类变量用众数。整个变换器活在 Pipeline 里，按折拟合。"""
    blocks = []
    if numeric:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scale", StandardScaler()))
        blocks.append(("num", Pipeline(steps), numeric))
    if categorical:
        blocks.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
        ]), categorical))
    return ColumnTransformer(blocks, remainder="drop")


def make_estimator(name, seed, n_pos, n_neg, epv, n_jobs=-1):
    """样本越紧张，模型越要简单。树的深度和数量随 EPV 收紧，
    因为复杂模型在小样本上记住的是噪声。"""
    balanced = min(n_pos, n_neg) / max(n_pos, n_neg) < 0.25
    tight = epv < 20
    if name == "lr":
        return LogisticRegression(
            C=0.5 if tight else 1.0, solver="lbfgs", max_iter=2000,
            class_weight="balanced" if balanced else None, random_state=seed), True
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, max_depth=3 if tight else 6,
            min_samples_leaf=max(5, int(0.02 * (n_pos + n_neg))),
            max_features="sqrt", class_weight="balanced_subsample" if balanced else None,
            random_state=seed, n_jobs=n_jobs), False
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=2 if tight else 4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, min_child_weight=max(3, int(0.02 * (n_pos + n_neg))),
            scale_pos_weight=(n_neg / n_pos) if balanced else 1.0,
            eval_metric="logloss", random_state=seed, n_jobs=n_jobs,
            tree_method="hist"), False
    raise ValueError(f"未知模型：{name}")


MODEL_LABELS = {"lr": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}


def build_model(name, numeric, categorical, seed, n_pos, n_neg, epv, calib_method, n_jobs=-1):
    est, needs_scaling = make_estimator(name, seed, n_pos, n_neg, epv, n_jobs)
    pipe = Pipeline([
        ("prep", build_preprocessor(numeric, categorical, needs_scaling)),
        ("clf", est),
    ])
    # 校准折数不能超过少数类例数，否则某些折里会没有阳性样本
    cv = int(min(5, max(2, min(n_pos, n_neg) // 5)))
    # ensemble=False：基模型在全部训练数据上训练一次，校准器在交叉验证的折外预测上拟合。
    # 默认的 ensemble=True 会把 k 个折模型的概率平均，这种平均把预测值向均值收缩，
    # 会人为制造出大于 1 的校准斜率，把真实的过拟合程度掩盖掉。顺带也快 k 倍。
    return CalibratedClassifierCV(pipe, method=calib_method, cv=cv, ensemble=False)


# ---------------------------------------------------------------- 内部验证

def apparent_performance(y, p):
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }


def _one_bootstrap(idx, spec, X, y):
    """单次 bootstrap 重抽样：重训模型，返回它在重抽样样本上和原始数据上的性能差。"""
    warnings.filterwarnings("ignore")   # loky 子进程不继承主进程的过滤器
    if len(np.unique(y[idx])) < 2:
        return None
    try:
        m = build_model(**spec, n_jobs=1)
        m.fit(X.iloc[idx], y[idx])
        p_in = m.predict_proba(X.iloc[idx])[:, 1]
        p_out = m.predict_proba(X)[:, 1]
        s_in, _ = calibration_slope_intercept(y[idx], p_in)
        s_out, _ = calibration_slope_intercept(y, p_out)
        return {
            "auc": roc_auc_score(y[idx], p_in) - roc_auc_score(y, p_out),
            "cal_slope": (s_in - s_out) if (np.isfinite(s_in) and np.isfinite(s_out)) else None,
        }
    except Exception:
        return None


def optimism_correction(spec, X, y, n_boot, rng, label, n_jobs=-1):
    """Harrell 的 bootstrap 乐观度校正。

    在原始数据上训练出来的模型，在同一批数据上评估必然偏乐观。做法是：重抽样一个
    bootstrap 样本，在上面重新训练，然后比较该模型在 bootstrap 样本上（乐观）和在
    原始数据上（相对客观）的表现，两者之差就是"乐观度"。平均 B 次后从表观性能里减掉。

    这比简单切分测试集更适合小样本——不浪费任何一例数据。

    重抽样索引先一次性生成，保证并行执行的结果与随机种子一一对应、可复现。
    各次重抽样彼此独立，所以按进程并行；内层模型退回单线程避免抢核。
    """
    y = np.asarray(y)
    n = len(y)
    all_idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    print(f"    [{label}] {n_boot} 次重抽样，并行度 {n_jobs} …", flush=True)
    out = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_one_bootstrap)(idx, spec, X, y) for idx in all_idx)

    aucs = [r["auc"] for r in out if r is not None]
    slopes = [r["cal_slope"] for r in out if r is not None and r["cal_slope"] is not None]
    return (
        {"auc": float(np.mean(aucs)) if aucs else 0.0,
         "cal_slope": float(np.mean(slopes)) if slopes else 0.0},
        len(aucs),
    )


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser(description="临床预测模型建模与内部验证")
    ap.add_argument("--data", required=True)
    ap.add_argument("--outcome", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--categorical", default="")
    ap.add_argument("--positive-label", default=None)
    ap.add_argument("--models", default="lr,rf,xgb", help="要跑的模型，逗号分隔：lr,rf,xgb")
    ap.add_argument("--strategy", default="auto", choices=["auto", "bootstrap", "split"])
    ap.add_argument("--calibration", default="auto", choices=["auto", "sigmoid", "isotonic"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=100,
                    help="乐观度校正的重抽样次数（每次都要重训模型）。100 次通常足够，追求稳定可加到 200")
    ap.add_argument("--n-jobs", type=int, default=-1, help="并行进程数，-1 表示用满所有核")
    ap.add_argument("--n-ci-boot", type=int, default=1000, help="置信区间的重抽样次数（不重训，快）")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ---- 数据准备（与质检脚本共用同一套判定逻辑，避免两处不一致）----
    df = load_table(args.data)
    if args.outcome not in df.columns:
        sys.exit(f"数据里没有列 '{args.outcome}'")
    y_ser, pos_label, _ = resolve_outcome(df, args.outcome, args.positive_label)
    keep = y_ser.notna()
    df, y_ser = df[keep].reset_index(drop=True), y_ser[keep].reset_index(drop=True)

    excluded = [c.strip() for c in args.exclude.split(",") if c.strip()]
    forced_cat = [c.strip() for c in args.categorical.split(",") if c.strip()]

    audit_path = out / "audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        numeric = [c for c in audit["predictors_numeric"] if c in df.columns]
        categorical = [c for c in audit["predictors_categorical"] if c in df.columns]
        epv_hint, auto_strategy = audit["epv"], audit["recommended_strategy"]
        print(f"沿用质检结果：{audit_path}")
    else:
        print("没找到 audit.json，就地重新判定变量类型（建议先跑 audit_data.py）")
        cand = [c for c in df.columns if c != args.outcome and c not in excluded]
        cand = [c for c in cand if df[c].nunique(dropna=True) > 1
                and df[c].isna().mean() <= 0.40]
        numeric, categorical, _ = split_types(df, cand, forced_cat)
        epv_hint, auto_strategy = None, None

    features = numeric + categorical
    if not features:
        sys.exit("没有可用的预测变量")
    X, y = df[features].copy(), y_ser.to_numpy(dtype=int)

    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    events = min(n_pos, n_neg)
    n_params = len(numeric) + sum(max(1, X[c].nunique(dropna=True) - 1) for c in categorical)
    epv = events / n_params if n_params else 0.0
    if epv_hint is not None:
        epv = epv_hint

    strategy = args.strategy
    if strategy == "auto":
        strategy = auto_strategy or ("bootstrap" if (epv < 10 or len(y) < 500) else "split")
    calib = args.calibration
    if calib == "auto":
        # isotonic 灵活但吃数据，小样本上自己就会过拟合；sigmoid 只有两个参数，更稳
        calib = "isotonic" if (events >= 100 and len(y) >= 1000) else "sigmoid"

    print(f"\n例数 {len(y)}｜阳性 {n_pos}｜阴性 {n_neg}｜EPV {epv:.1f}")
    print(f"验证策略：{strategy}｜概率校准：{calib}｜随机种子：{args.seed}")
    print(f"预测变量 {len(features)} 个（连续 {len(numeric)}、分类 {len(categorical)}）\n")

    if strategy == "split":
        test_size = 0.25 if len(y) >= 2000 else 0.30
        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=args.seed)
        eval_label = f"独立验证集（{len(y_ev)} 例）"
    else:
        X_tr, y_tr, X_ev, y_ev = X, y, X, y
        test_size = None
        eval_label = f"全数据表观性能（{len(y)} 例，已做 bootstrap 乐观度校正）"

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results, fitted, preds = {}, {}, {}

    for name in model_names:
        print(f"训练 {MODEL_LABELS.get(name, name)} …", flush=True)
        spec = {
            "name": name, "numeric": numeric, "categorical": categorical,
            "seed": args.seed, "n_pos": int(y_tr.sum()),
            "n_neg": int(len(y_tr) - y_tr.sum()), "epv": epv, "calib_method": calib,
        }
        model = build_model(**spec)
        model.fit(X_tr, y_tr)
        p_ev = model.predict_proba(X_ev)[:, 1]

        perf = apparent_performance(y_ev, p_ev)
        lo, hi = auc_ci_bootstrap(y_ev, p_ev, args.n_ci_boot, rng)
        thr = youden_threshold(y_ev, p_ev)
        hl_stat, hl_p = hosmer_lemeshow(y_ev, p_ev)

        entry = {
            "model": name,
            "label": MODEL_LABELS.get(name, name),
            "n_eval": int(len(y_ev)),
            "auc": perf["auc"], "auc_ci_low": lo, "auc_ci_high": hi,
            "brier": perf["brier"],
            "cal_slope": perf["cal_slope"], "cal_intercept": perf["cal_intercept"],
            "hl_stat": hl_stat, "hl_p": hl_p,
            "mean_predicted_risk": float(np.mean(p_ev)),
            "observed_risk": float(np.mean(y_ev)),
            **{f"opt_{k}": v for k, v in threshold_metrics(y_ev, p_ev, thr).items()},
        }

        gaps, n_ok = optimism_correction(spec, X_tr, y_tr, args.n_boot, rng, name, args.n_jobs)
        entry["optimism_auc"] = gaps["auc"]
        entry["optimism_cal_slope"] = gaps["cal_slope"]
        entry["auc_corrected"] = perf["auc"] - gaps["auc"]
        entry["cal_slope_corrected"] = perf["cal_slope"] - gaps["cal_slope"]
        entry["n_boot_successful"] = n_ok

        results[name] = entry
        fitted[name] = model
        preds[name] = p_ev
        print(f"  AUC {perf['auc']:.3f} (95%CI {lo:.3f}-{hi:.3f}) → 校正后 {entry['auc_corrected']:.3f}"
              f"｜校准斜率 {perf['cal_slope']:.2f} → 校正后 {entry['cal_slope_corrected']:.2f}"
              f"｜Brier {perf['brier']:.3f}\n", flush=True)

    # ---- 模型间比较 ----
    comparisons = []
    for i, a in enumerate(model_names):
        for b in model_names[i + 1:]:
            diff, pval = delong_test(y_ev, preds[a], preds[b])
            comparisons.append({
                "model_a": MODEL_LABELS.get(a, a), "model_b": MODEL_LABELS.get(b, b),
                "auc_difference": diff, "delong_p": pval,
            })

    best = max(results, key=lambda k: results[k]["auc_corrected"])

    # ---- 决策曲线 ----
    thresholds = np.arange(0.01, 0.95, 0.01)
    dca_rows = []
    for name in model_names:
        for row in net_benefit_curve(y_ev, preds[name], thresholds):
            dca_rows.append({"model": MODEL_LABELS.get(name, name), **row})
    pd.DataFrame(dca_rows).to_csv(out / "dca.csv", index=False)

    pred_df = pd.DataFrame({"y_true": y_ev})
    for name in model_names:
        pred_df[MODEL_LABELS.get(name, name)] = preds[name]
    pred_df.to_csv(out / "predictions.csv", index=False)

    # ---- 自动问题检出：这些是审稿人最先看的地方 ----
    findings = []

    def add(level, msg, action):
        findings.append({"level": level, "message": msg, "action": action})

    b = results[best]
    # bootstrap 策略下评估集就是训练集，表观校准斜率会被 Platt 校准系统性抬高
    # （校准器在折外预测上拟合，却应用到更容易区分的折内预测上），所以判断要看校正后的值
    use_corrected = strategy == "bootstrap"
    slope_used = b["cal_slope_corrected"] if use_corrected else b["cal_slope"]
    slope_name = "校正后校准斜率" if use_corrected else "校准斜率"

    if b["auc_corrected"] < 0.70:
        add("critical",
            f"最优模型校正后 AUC 仅 {b['auc_corrected']:.3f}，区分能力不足。",
            "现有变量携带的信息不够。考虑补充更有预测力的临床指标，或重新检视结局定义是否过于宽泛、人群是否过于异质。")
    if b["optimism_auc"] > 0.05:
        add("critical",
            f"乐观度达 {b['optimism_auc']:.3f}（表观 AUC {b['auc']:.3f} → 校正后 {b['auc_corrected']:.3f}），过拟合明显。",
            "减少变量个数或进一步简化模型。论文里必须报告校正后的数值，只报表观 AUC 会被审稿人指出。")
    if np.isfinite(slope_used) and not (0.8 <= slope_used <= 1.2):
        direction = ("预测值过于分散——高风险的估得太高、低风险的估得太低，是过拟合的典型表现"
                     if slope_used < 1
                     else "预测值过于集中在患病率附近，模型没有充分拉开风险差距")
        add("critical" if slope_used < 0.7 else "warning",
            f"{slope_name} {slope_used:.2f}，偏离理想值 1.0：{direction}。",
            "这是审稿人最爱抓的点。可以尝试加大正则化强度或减少变量个数。")

    # 区分度最好的模型未必校准最好。临床上一个校准差的模型会系统性误导风险判断，
    # 所以当两者不是同一个模型、且 AUC 差距不大时，值得提醒研究者重新权衡。
    slope_key = "cal_slope_corrected" if use_corrected else "cal_slope"
    calib_ranked = [m for m in results.values() if np.isfinite(results[m["model"]][slope_key])]
    if len(calib_ranked) > 1:
        best_calib = min(calib_ranked, key=lambda m: abs(m[slope_key] - 1.0))
        auc_gap = b["auc_corrected"] - best_calib["auc_corrected"]
        if best_calib["model"] != best and auc_gap < 0.05:
            add("warning",
                f"{b['label']} 区分度最好（校正后 AUC {b['auc_corrected']:.3f}，{slope_name} {slope_used:.2f}），"
                f"但 {best_calib['label']} 校准更好（{slope_name} {best_calib[slope_key]:.2f}，"
                f"校正后 AUC {best_calib['auc_corrected']:.3f}，仅低 {auc_gap:.3f}）。",
                f"临床预测模型的价值在于给出可信的风险数值，校准差的模型会系统性误导医生的风险判断。"
                f"区分度只差 {auc_gap:.3f} 的情况下，优先考虑 {best_calib['label']}——它还更容易解释和推广。")
    if abs(b["mean_predicted_risk"] - b["observed_risk"]) > 0.05:
        add("warning",
            f"平均预测风险 {b['mean_predicted_risk']:.1%} 与实际发生率 {b['observed_risk']:.1%} 相差较大。",
            "整体风险水平存在系统性偏移，在论文中需说明，外推到其他人群时要重新校准截距。")
    lr_res = results.get("lr")
    if lr_res and best != "lr" and (b["auc_corrected"] - lr_res["auc_corrected"]) < 0.02:
        add("info",
            f"{b['label']} 校正后 AUC {b['auc_corrected']:.3f}，仅比 Logistic 回归的 {lr_res['auc_corrected']:.3f} 高 {b['auc_corrected'] - lr_res['auc_corrected']:.3f}。",
            "机器学习没有带来实质提升。诚实的写法是承认传统回归已经够用——这本身是有价值的结论，而且 Logistic 模型更容易解释和推广，临床上反而更受欢迎。")
    if strategy == "bootstrap" and len(comparisons) > 0:
        add("info",
            "模型间的 DeLong 检验是在训练数据上做的，会系统性偏袒更复杂的模型（复杂模型更善于记住训练数据）。",
            "论文中比较模型优劣时，应以 bootstrap 校正后的 AUC 为准，DeLong 的 p 值仅供参考，或改在独立验证集上计算。")
    if strategy == "bootstrap":
        add("info",
            "样本量不支持切分独立验证集，性能来自全数据表观值 + bootstrap 乐观度校正。",
            "论文中必须写明这一点，并强调仍需外部队列验证。这是内部验证，不能替代外部验证。")
    if events < 50:
        add("warning",
            f"事件数仅 {events} 例，所有指标的置信区间都会很宽。",
            "结论要写得保守，明确说明这是探索性研究，需要更大样本验证。")

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_file": str(args.data),
        "outcome": args.outcome,
        "positive_label": str(pos_label),
        "n_total": int(len(y)), "n_positive": n_pos, "n_negative": n_neg,
        "prevalence": float(y.mean()),
        "epv": round(float(epv), 2),
        "strategy": strategy, "eval_set_description": eval_label,
        "test_size": test_size,
        "calibration_method": calib,
        "seed": args.seed, "n_boot": args.n_boot,
        "features_numeric": numeric, "features_categorical": categorical,
        "models": results,
        "model_comparisons": comparisons,
        "best_model": best,
        "best_model_label": MODEL_LABELS.get(best, best),
        "calibration_slope_to_report": "cal_slope_corrected" if use_corrected else "cal_slope",
        "findings": findings,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    cols = ["label", "n_eval", "auc", "auc_ci_low", "auc_ci_high", "auc_corrected",
            "optimism_auc", "cal_slope", "cal_slope_corrected", "cal_intercept",
            "brier", "hl_p", "opt_threshold", "opt_sensitivity", "opt_specificity",
            "opt_PPV", "opt_NPV", "opt_accuracy", "opt_F1"]
    pd.DataFrame([{c: results[m].get(c) for c in cols} for m in model_names]).to_csv(
        out / "metrics.csv", index=False)

    # ---- 模型包：一个文件装下预处理 + 模型，下游直接喂原始 DataFrame ----
    joblib.dump({"models": fitted, "best": best, "features": features}, out / "model_bundle.pkl")
    joblib.dump({"X": X_ev, "y": y_ev, "features": features}, out / "eval_data.pkl")

    card = {
        "best_model": MODEL_LABELS.get(best, best),
        "outcome": args.outcome, "positive_label": str(pos_label),
        "feature_order": features,
        "numeric_features": numeric, "categorical_features": categorical,
        "categorical_levels": {c: [str(v) for v in sorted(X[c].dropna().unique(), key=str)]
                               for c in categorical},
        "numeric_ranges": {c: {"min": float(X[c].min()), "max": float(X[c].max()),
                               "median": float(X[c].median())} for c in numeric},
        "training_n": int(len(y_tr)), "seed": args.seed,
        "calibration_method": calib, "validation_strategy": strategy,
        "auc_corrected": results[best]["auc_corrected"],
        "usage": "joblib.load('model_bundle.pkl')['models'][best].predict_proba(df[feature_order])[:,1]",
        "disclaimer": "仅供科研与临床辅助参考，不能替代医师判断。",
        "software": {
            "python": platform.python_version(),
            "scikit_learn": __import__("sklearn").__version__,
            "pandas": pd.__version__, "numpy": np.__version__,
        },
    }
    (out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"最优模型：{MODEL_LABELS.get(best, best)}")
    print(f"  校正后 AUC {results[best]['auc_corrected']:.3f}"
          f"｜校正后校准斜率 {results[best]['cal_slope_corrected']:.2f}"
          f"｜Brier {results[best]['brier']:.3f}")
    print(f"\n结果已写入 {out}/")
    if findings:
        print("\n自动检出的问题：")
        for f in findings:
            print(f"  [{f['level']}] {f['message']}")
    print("\n下一步：python scripts/make_plots.py --results", out)


if __name__ == "__main__":
    main()
