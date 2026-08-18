#!/usr/bin/env python3
"""临床预测模型 —— 第 3 步：期刊级图表。

图内文字一律英文，直接投稿用。每张图同时输出 300 dpi TIFF（多数期刊要求的位图格式）
和矢量 PDF（放大不糊，排版编辑更喜欢）。
"""
import argparse
import json
import logging
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

warnings.filterwarnings("ignore")
# 部分中文字体只提供 weight 500，matplotlib 会对每个文本元素刷一条 findfont 提示，纯噪音
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# 无彩色打印和色觉障碍读者也能区分的配色
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9"]

# 中文变量名如果没有对应字形，matplotlib 会静默画成一串方框——图看起来"生成成功"，
# 实际完全不能用。这里主动找一个能显示中文的字体挂到 fallback 链上。
CJK_CANDIDATES = ["Noto Sans CJK SC", "Source Han Sans SC", "WenQuanYi Zen Hei",
                  "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC", "Arial Unicode MS"]


def find_cjk_font():
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for name in CJK_CANDIDATES:
        if name in available:
            return name
    return None


CJK_FONT = find_cjk_font()
_SANS = ["DejaVu Sans", "Arial", "Helvetica"]
if CJK_FONT:
    _SANS.insert(0, CJK_FONT)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": _SANS,
    "axes.unicode_minus": False,
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "lines.linewidth": 1.6,
    "figure.dpi": 100, "savefig.bbox": "tight",
})


def load_labels(results_dir, label_file):
    """读取变量名中英对照表。投英文期刊时图里必须是英文术语，中文变量名直接上图是不能用的。"""
    path = Path(label_file) if label_file else (results_dir / "labels.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_label_template(results_dir, card, raw_names):
    """自动生成对照表模板，用户只要把右边填成英文再重跑一次即可。

    模板里既包含变量名，也包含分类变量的各个水平——独热编码后水平名会出现在图上。
    """
    mapping = {n: n for n in raw_names}
    for levels in (card.get("categorical_levels") or {}).values():
        for lv in levels:
            mapping.setdefault(str(lv), str(lv))
    path = results_dir / "labels_template.json"
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def apply_labels(name, mapping):
    """按最长键优先做替换，这样 '心律失常' 不会被 '心律' 抢先替掉。

    独热编码后的名字形如 '病因分型_暴发性心肌炎'，变量名和水平名都需要替换，
    所以做的是子串替换而不是整名查表。
    """
    if not mapping:
        return name
    for key in sorted(mapping, key=len, reverse=True):
        if key and key in name:
            name = name.replace(key, str(mapping[key]))
    return name


def save(fig, outdir, stem):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.tiff", dpi=300, format="tiff",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(outdir / f"{stem}.pdf", format="pdf")
    plt.close(fig)
    print(f"  {stem}.tiff / .pdf")


def plot_roc(preds, metrics, outdir):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    y = preds["y_true"].to_numpy()
    for i, (key, m) in enumerate(metrics["models"].items()):
        label = m["label"]
        if label not in preds.columns:
            continue
        fpr, tpr, _ = roc_curve(y, preds[label])
        ax.plot(fpr, tpr, color=PALETTE[i % len(PALETTE)],
                label=f"{label}\nAUC = {m['auc']:.3f} ({m['auc_ci_low']:.3f}–{m['auc_ci_high']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.0)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Receiver Operating Characteristic")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)
    save(fig, outdir, "fig1_roc")


def plot_calibration(preds, metrics, outdir):
    y = preds["y_true"].to_numpy()
    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(4.2, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.plot([0, 1], [0, 1], "--", color="#999999", lw=1.0, label="Ideal")
    n_bins = 10 if len(y) >= 200 else 5
    for i, (key, m) in enumerate(metrics["models"].items()):
        label = m["label"]
        if label not in preds.columns:
            continue
        p = preds[label].to_numpy()
        try:
            frac, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        except Exception:
            continue
        c = PALETTE[i % len(PALETTE)]
        leg = f"{label}\nslope = {m['cal_slope']:.2f}"
        if metrics.get("calibration_slope_to_report") == "cal_slope_corrected":
            leg += f" (corrected {m['cal_slope_corrected']:.2f})"
        leg += f", Brier = {m['brier']:.3f}"
        ax.plot(mean_pred, frac, "o-", color=c, ms=4, label=leg)
        axh.hist(p, bins=25, range=(0, 1), histtype="step", color=c, lw=1.0)
    ax.set_ylabel("Observed proportion")
    title = "Calibration"
    if metrics.get("strategy") == "bootstrap":
        # 曲线画的是训练数据上的表观校准，不标注会让读者以为这是独立验证结果
        title += " (apparent, training set)"
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # 校准曲线沿对角线走，右下角总是空的，图例放这里不会压住数据
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)
    axh.set_xlabel("Predicted probability")
    axh.set_ylabel("Count")
    axh.set_yticks([])
    save(fig, outdir, "fig2_calibration")


def plot_dca(results_dir, metrics, outdir):
    path = results_dir / "dca.csv"
    if not path.exists():
        return
    dca = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    first = dca[dca["model"] == dca["model"].iloc[0]]
    ax.plot(first["threshold"], first["net_benefit_all"], "--",
            color="#666666", lw=1.1, label="Treat all")
    ax.axhline(0, color="#000000", lw=0.9, label="Treat none")
    for i, name in enumerate(dca["model"].unique()):
        sub = dca[dca["model"] == name]
        ax.plot(sub["threshold"], sub["net_benefit_model"],
                color=PALETTE[i % len(PALETTE)], label=name)
    prev = metrics.get("prevalence", 0.1)
    ax.set_xlim(0, min(0.9, max(0.4, prev * 4)))
    ymax = float(dca["net_benefit_model"].max())
    ax.set_ylim(-0.05 * max(ymax, 0.05), ymax * 1.25 if ymax > 0 else 0.1)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend(loc="upper right", frameon=False, handlelength=1.4)
    save(fig, outdir, "fig3_decision_curve")


def _unwrap(calibrated):
    """从 CalibratedClassifierCV 里取出第一折的底层 Pipeline。

    校准器本身是个包装，SHAP 要的是里面真正的树模型。用第一折的模型做解释是
    通行做法——各折之间的特征重要性通常高度一致。
    """
    inner = getattr(calibrated, "calibrated_classifiers_", None)
    if not inner:
        return None
    return getattr(inner[0], "estimator", None)


def plot_importance(results_dir, metrics, outdir, mapping):
    bundle_path = results_dir / "model_bundle.pkl"
    data_path = results_dir / "eval_data.pkl"
    if not (bundle_path.exists() and data_path.exists()):
        return
    bundle = joblib.load(bundle_path)
    data = joblib.load(data_path)
    best_key = bundle["best"]
    pipe = _unwrap(bundle["models"][best_key])
    if pipe is None:
        print("  （无法取出底层模型，跳过重要性图）")
        return

    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    X = data["X"]
    Xt = prep.transform(X)
    names = [apply_labels(n.split("__", 1)[-1], mapping) for n in prep.get_feature_names_out()]
    label = metrics["best_model_label"]

    if best_key == "lr":
        # 逻辑回归的自然表达是比值比森林图，比 SHAP 更贴合临床阅读习惯
        coef = clf.coef_[0]
        order = np.argsort(np.abs(coef))[-20:]
        or_vals = np.exp(coef[order])
        fig, ax = plt.subplots(figsize=(4.6, max(3.0, 0.26 * len(order) + 1.0)))
        ypos = np.arange(len(order))
        ax.scatter(or_vals, ypos, color=PALETTE[0], s=22, zorder=3)
        ax.hlines(ypos, 1, or_vals, color=PALETTE[0], lw=1.2, zorder=2)
        ax.axvline(1, color="#666666", ls="--", lw=1.0)
        ax.set_yticks(ypos)
        ax.set_yticklabels([names[i] for i in order])
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (per 1 SD, log scale)")
        ax.set_title(f"{label}: adjusted odds ratios")
        save(fig, outdir, "fig4_importance")
        return

    try:
        import shap
        n_bg = min(len(Xt), 500)
        idx = np.random.default_rng(metrics.get("seed", 42)).choice(len(Xt), n_bg, replace=False)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xt[idx])
        if isinstance(sv, list):
            sv = sv[1]
        elif sv.ndim == 3:
            sv = sv[:, :, 1]

        fig = plt.figure(figsize=(5.2, max(3.2, 0.26 * min(len(names), 20) + 1.2)))
        shap.summary_plot(sv, Xt[idx], feature_names=names, max_display=20, show=False)
        plt.title(f"{label}: SHAP value distribution", fontsize=11)
        plt.xlabel("SHAP value (impact on predicted log-odds)")
        save(plt.gcf(), outdir, "fig4_shap_beeswarm")

        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[-20:]
        fig, ax = plt.subplots(figsize=(4.6, max(3.0, 0.26 * len(order) + 1.0)))
        ax.barh(np.arange(len(order)), mean_abs[order], color=PALETTE[0], height=0.65)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([names[i] for i in order])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"{label}: feature importance")
        save(fig, outdir, "fig5_shap_importance")
    except Exception as exc:
        print(f"  （SHAP 计算失败，退回到内置重要性：{exc}）")
        imp = getattr(clf, "feature_importances_", None)
        if imp is None:
            return
        order = np.argsort(imp)[-20:]
        fig, ax = plt.subplots(figsize=(4.6, max(3.0, 0.26 * len(order) + 1.0)))
        ax.barh(np.arange(len(order)), imp[order], color=PALETTE[0], height=0.65)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([names[i] for i in order])
        ax.set_xlabel("Feature importance")
        ax.set_title(f"{label}: feature importance")
        save(fig, outdir, "fig4_importance")


def main():
    ap = argparse.ArgumentParser(description="生成期刊级图表")
    ap.add_argument("--results", default="results", help="run_pipeline.py 的输出目录")
    ap.add_argument("--labels", default=None,
                    help="变量名中英对照表 JSON。不指定时会自动读取输出目录下的 labels.json")
    args = ap.parse_args()

    rd = Path(args.results)
    mpath = rd / "metrics.json"
    if not mpath.exists():
        raise SystemExit(f"找不到 {mpath}，请先运行 run_pipeline.py")
    metrics = json.loads(mpath.read_text(encoding="utf-8"))
    preds = pd.read_csv(rd / "predictions.csv")
    outdir = rd / "figures"

    mapping = load_labels(rd, args.labels)
    raw_names = metrics.get("features_numeric", []) + metrics.get("features_categorical", [])
    has_cjk = any(ord(ch) > 127 for n in raw_names for ch in n)

    if has_cjk and not mapping:
        card_path = rd / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        tpl = write_label_template(rd, card, raw_names)
        print(f"⚠ 变量名里有中文，但没找到中英对照表。")
        print(f"  已生成模板：{tpl}")
        print(f"  把每行右边改成英文术语，另存为 {rd}/labels.json，再重跑本脚本，")
        print(f"  图里就会是英文——投英文期刊必须这样做。")
        if CJK_FONT:
            print(f"  本次先用中文字体 {CJK_FONT} 出图，可以先看效果。\n")
        else:
            print(f"  ⚠ 而且当前环境没有中文字体，中文会显示成方框，这次的图不能用。\n")
    elif has_cjk and not CJK_FONT and len(mapping) < len(raw_names):
        print(f"⚠ 对照表没有覆盖全部变量，未覆盖的中文会显示成方框（环境缺中文字体）。\n")

    print("生成图表：")
    plot_roc(preds, metrics, outdir)
    plot_calibration(preds, metrics, outdir)
    plot_dca(rd, metrics, outdir)
    plot_importance(rd, metrics, outdir, mapping)
    print(f"\n全部图已保存到 {outdir}/（300 dpi TIFF + 矢量 PDF）")


if __name__ == "__main__":
    main()
