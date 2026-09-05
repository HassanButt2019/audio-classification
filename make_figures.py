import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGDIR = Path("figures")
MODELS = ["cnn", "crnn", "vggish"]
MODEL_LABEL = {"cnn": "CNN", "crnn": "CRNN", "vggish": "VGGish"}
CONDITIONS = ["normal", "adv_train_fgsm", "adv_train_bim", "adv_finetune_fgsm", "adv_finetune_bim"]
COND_LABEL = {
    "normal": "Baseline",
    "adv_train_fgsm": "AT-FGSM",
    "adv_train_bim": "AT-BIM",
    "adv_finetune_fgsm": "AFT-FGSM",
    "adv_finetune_bim": "AFT-BIM",
}
EPS = ["eps_0.01", "eps_0.03", "eps_0.1"]
EPS_VAL = [0.01, 0.03, 0.10]
ATTACKS = ["fgsm", "bim"]

COND_COLOR = dict(zip(CONDITIONS, ["#000000", "#0072B2", "#E69F00", "#009E73", "#CC79A7"]))
COND_HATCH = dict(zip(CONDITIONS, ["", "//", "\\\\", "xx", ".."]))
COND_MARKER = dict(zip(CONDITIONS, ["o", "s", "^", "D", "v"]))
MODEL_MARKER = dict(zip(MODELS, ["o", "s", "^"]))

plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "legend.fontsize": 9,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "figure.dpi": 300})


def warn(msg):
    print(f"WARNING: {msg}")


def to_percent(x):
    return x * 100.0 if x is not None and x <= 1.5 else x


def cv_path(model, cond):
    hits = sorted((RESULTS / model / cond).rglob("cv_summary.json"))
    return hits[0] if hits else None


def load_cv():
    out = {}
    for m in MODELS:
        for c in CONDITIONS:
            p = cv_path(m, c)
            if p is None:
                warn(f"no cv_summary.json for {m}/{c}")
                continue
            out[(m, c)] = json.loads(p.read_text())
    return out


CV = load_cv()


def clean_acc(m, c):
    d = CV.get((m, c))
    if not d or "accuracy" not in d.get("mean", {}):
        return None
    return to_percent(d["mean"]["accuracy"])


def clean_std(m, c):
    d = CV.get((m, c))
    if not d or "accuracy" not in d.get("std", {}):
        return None
    v = d["std"]["accuracy"]
    return v * 100.0 if to_percent(d["mean"]["accuracy"]) != d["mean"]["accuracy"] else v


def f1_macro(m, c):
    d = CV.get((m, c))
    if not d or "f1_macro" not in d.get("mean", {}):
        return None
    return to_percent(d["mean"]["f1_macro"])


def robust_folds(m, c, attack, eps):
    d = CV.get((m, c))
    if not d or attack not in d:
        return []
    vals = []
    for k in sorted(d[attack]):
        v = d[attack][k].get(eps)
        if v is not None:
            vals.append(v)
    return vals


def robust_mean(m, c, attack, eps):
    v = robust_folds(m, c, attack, eps)
    return float(np.mean(v)) if v else None


def mean_robustness_per_fold(m, c):
    d = CV.get((m, c))
    if not d or not all(a in d for a in ATTACKS):
        return []
    folds = sorted(set(d["fgsm"]) & set(d["bim"]))
    out = []
    for f in folds:
        cells = [d[a][f][e] for a in ATTACKS for e in EPS if e in d[a][f]]
        if len(cells) == 6:
            out.append(float(np.mean(cells)))
    return out


def history(m, c):
    root = RESULTS / m / c
    files = sorted(root.rglob("fold_*_results.json"))
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    for p in files:
        h = json.loads(p.read_text()).get("training_history", {})
        for k in hist:
            if k in h:
                hist[k].append(h[k])
    return hist


def mean_curve(runs):
    if not runs:
        return None
    n = min(len(r) for r in runs)
    return np.mean([r[:n] for r in runs], axis=0)


def epochs(m, c):
    runs = history(m, c)["train_loss"]
    return float(np.mean([len(r) for r in runs])) if runs else None


def save(fig, name, report):
    FIGDIR.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGDIR / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n=== {name} ===")
    for line in report:
        print("  " + line)


def fig_4_1():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.08
    rep = []
    for j, c in enumerate(CONDITIONS):
        accs, errs, f1s, xa, xf = [], [], [], [], []
        for i, m in enumerate(MODELS):
            a, s, f = clean_acc(m, c), clean_std(m, c), f1_macro(m, c)
            if a is None:
                warn(f"fig_4_1: missing accuracy {m}/{c}")
            else:
                accs.append(a); errs.append(s if s is not None else 0.0); xa.append(i - 0.45 + j * w + w / 2)
            if f is None:
                warn(f"fig_4_1: missing f1_macro {m}/{c}")
            else:
                f1s.append(f); xf.append(i + 0.05 + j * w + w / 2)
            rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]}: acc={a:.2f}±{s:.2f}%  f1={f:.2f}%" if None not in (a, s, f) else f"{MODEL_LABEL[m]} {COND_LABEL[c]}: incomplete")
        ax.bar(xa, accs, w, yerr=errs, capsize=2, color=COND_COLOR[c], hatch=COND_HATCH[c],
               edgecolor="white", label=COND_LABEL[c])
        ax.bar(xf, f1s, w, color=COND_COLOR[c], hatch=COND_HATCH[c], edgecolor="white", alpha=0.55)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{MODEL_LABEL[m]}\n(left: accuracy, right: macro F1)" for m in MODELS])
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    save(fig, "fig_4_1_model_performance.png", rep)


def fig_4_2():
    conds = ["normal", "adv_train_bim"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    rep = []
    for i, m in enumerate(MODELS):
        for c in conds:
            h = history(m, c)
            for row, (tr, va, ylab) in enumerate([("train_acc", "val_acc", "Accuracy (%)"),
                                                  ("train_loss", "val_loss", "Loss")]):
                ax = axes[row][i]
                for key, ls in ((tr, "-"), (va, "--")):
                    curve = mean_curve(h[key])
                    if curve is None:
                        warn(f"fig_4_2: missing {key} for {m}/{c}")
                        continue
                    ax.plot(np.arange(1, len(curve) + 1), curve, ls, color=COND_COLOR[c],
                            marker=COND_MARKER[c], markevery=max(1, len(curve) // 6), markersize=4,
                            label=f"{COND_LABEL[c]} {'train' if key.startswith('train') else 'val'}")
                    rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]} {key}: first={curve[0]:.3f} last={curve[-1]:.3f} n_epochs={len(curve)}")
                ax.set_ylabel(ylab)
        axes[0][i].set_title(MODEL_LABEL[m], fontsize=11)
        axes[1][i].set_xlabel("Epoch")
    axes[0][0].legend(fontsize=8, frameon=False)
    axes[1][0].legend(fontsize=8, frameon=False)
    save(fig, "fig_4_2_training_curves.png", rep)


def fig_4_4():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    rep = []
    for i, m in enumerate(MODELS):
        for r, a in enumerate(ATTACKS):
            ax = axes[r][i]
            for c in CONDITIONS:
                ys = [robust_mean(m, c, a, e) for e in EPS]
                if any(y is None for y in ys):
                    warn(f"fig_4_4: missing {a} values for {m}/{c}")
                    continue
                ax.plot(EPS_VAL, ys, marker=COND_MARKER[c], color=COND_COLOR[c], label=COND_LABEL[c])
                rep.append(f"{MODEL_LABEL[m]} {a.upper()} {COND_LABEL[c]}: " + ", ".join(f"eps={e}: {y:.2f}%" for e, y in zip(EPS_VAL, ys)))
            ax.set_ylabel(f"{a.upper()} robust accuracy (%)")
        axes[0][i].set_title(MODEL_LABEL[m], fontsize=11)
        axes[1][i].set_xlabel("Epsilon (L-inf)")
    axes[0][0].legend(fontsize=8, frameon=False)
    save(fig, "fig_4_4_accuracy_vs_epsilon.png", rep)


def fig_4_5():
    fig, ax = plt.subplots(figsize=(7, 5))
    rep = []
    cleans, robusts = [], []
    for m in MODELS:
        cl = clean_acc(m, "normal")
        rb = robust_mean(m, "adv_train_bim", "bim", "eps_0.01")
        if cl is None or rb is None:
            warn(f"fig_4_5: missing clean or AT-BIM eps=0.01 robust accuracy for {m}")
            continue
        ax.plot([0, 1], [cl, rb], marker=MODEL_MARKER[m], color=COND_COLOR[CONDITIONS[MODELS.index(m)]],
                linewidth=2, markersize=9, label=MODEL_LABEL[m])
        cleans.append(cl); robusts.append(rb)
        rep.append(f"{MODEL_LABEL[m]}: clean={cl:.2f}% -> AT-BIM BIM eps=0.01 robust={rb:.2f}%")
    if cleans:
        rep.append(f"clean spread = {max(cleans)-min(cleans):.2f} pp; robust spread = {max(robusts)-min(robusts):.2f} pp")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Clean accuracy\n(Baseline)", "Robust accuracy\n(AT-BIM, BIM, $\\epsilon$=0.01)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(-0.3, 1.3)
    ax.legend(frameon=False)
    save(fig, "fig_4_5_robustness_convergence.png", rep)


def fig_4_6():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    rep = []
    cells = [(a, e, ev) for a in ATTACKS for e, ev in zip(EPS, EPS_VAL)]
    w = 0.16
    for i, m in enumerate(MODELS):
        ax = axes[i]
        for j, c in enumerate(CONDITIONS):
            xs, ys = [], []
            for k, (a, e, ev) in enumerate(cells):
                v = robust_mean(m, c, a, e)
                if v is None:
                    warn(f"fig_4_6: missing {a} {e} for {m}/{c}")
                    continue
                xs.append(k - 0.4 + j * w + w / 2); ys.append(100.0 - v)
                rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]} {a.upper()} eps={ev}: ASR={100.0-v:.2f}%")
            ax.bar(xs, ys, w, color=COND_COLOR[c], hatch=COND_HATCH[c], edgecolor="white", label=COND_LABEL[c])
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{a.upper()}\n{ev}" for a, _, ev in cells], fontsize=8)
        ax.set_xlabel("Attack and epsilon (L-inf)")
        ax.set_title(MODEL_LABEL[m], fontsize=11)
    axes[0].set_ylabel("Attack success rate (%)")
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, "fig_4_6_attack_success_rate.png", rep)


def fig_4_8():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rep = []
    for m in MODELS:
        for c in CONDITIONS:
            ep = epochs(m, c)
            folds = mean_robustness_per_fold(m, c)
            if ep is None or not folds:
                warn(f"fig_4_8: missing epochs or robustness for {m}/{c}")
                continue
            rb = float(np.mean(folds))
            ax.scatter(ep, rb, marker=MODEL_MARKER[m], color=COND_COLOR[c], s=90,
                       edgecolor="black", linewidth=0.5, zorder=3)
            ax.annotate(COND_LABEL[c], (ep, rb), textcoords="offset points", xytext=(6, 4), fontsize=8)
            rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]}: epochs={ep:.1f}  mean robustness={rb:.2f}%")
    handles = [plt.Line2D([], [], marker=MODEL_MARKER[m], color="grey", linestyle="", label=MODEL_LABEL[m]) for m in MODELS]
    handles += [plt.Line2D([], [], marker="s", color=COND_COLOR[c], linestyle="", label=COND_LABEL[c]) for c in CONDITIONS]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2)
    ax.set_xlabel("Training epochs")
    ax.set_ylabel("Mean robust accuracy (%)")
    save(fig, "fig_4_8_cost_vs_robustness.png", rep)


def fig_4_9():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rep = []
    w = 0.16
    for j, c in enumerate(CONDITIONS):
        xs, ys, es = [], [], []
        for i, m in enumerate(MODELS):
            folds = mean_robustness_per_fold(m, c)
            if not folds:
                warn(f"fig_4_9: missing per-fold robustness for {m}/{c}")
                continue
            xs.append(i - 0.4 + j * w + w / 2); ys.append(float(np.mean(folds))); es.append(float(np.std(folds)))
            rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]}: mean robustness={ys[-1]:.2f}% ± {es[-1]:.2f} (n_folds={len(folds)})")
        ax.bar(xs, ys, w, yerr=es, capsize=2, color=COND_COLOR[c], hatch=COND_HATCH[c],
               edgecolor="white", label=COND_LABEL[c])
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABEL[m] for m in MODELS])
    ax.set_xlabel("Model")
    ax.set_ylabel("Mean robust accuracy (%)")
    ax.legend(fontsize=9, frameon=False, ncol=3)
    save(fig, "fig_4_9_robustness_across_models.png", rep)


def fig_4_10():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rep = []
    for m in MODELS:
        for c in CONDITIONS:
            cl = clean_acc(m, c)
            folds = mean_robustness_per_fold(m, c)
            if cl is None or not folds:
                warn(f"fig_4_10: missing clean accuracy or robustness for {m}/{c}")
                continue
            rb = float(np.mean(folds))
            ax.scatter(cl, rb, marker=MODEL_MARKER[m], color=COND_COLOR[c], s=90,
                       edgecolor="black", linewidth=0.5, zorder=3)
            rep.append(f"{MODEL_LABEL[m]} {COND_LABEL[c]}: clean={cl:.2f}%  mean robustness={rb:.2f}%")
    handles = [plt.Line2D([], [], marker=MODEL_MARKER[m], color="grey", linestyle="", label=MODEL_LABEL[m]) for m in MODELS]
    handles += [plt.Line2D([], [], marker="s", color=COND_COLOR[c], linestyle="", label=COND_LABEL[c]) for c in CONDITIONS]
    ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2)
    ax.set_xlabel("Clean accuracy (%)")
    ax.set_ylabel("Mean robust accuracy (%)")
    save(fig, "fig_4_10_clean_vs_robust.png", rep)


if __name__ == "__main__":
    for f in (fig_4_1, fig_4_2, fig_4_4, fig_4_5, fig_4_6, fig_4_8, fig_4_9, fig_4_10):
        f()
