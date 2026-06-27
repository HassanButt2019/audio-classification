import streamlit as st
import json
import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(ROOT, "results")

MODES = {
    "normal": "Normal Training",
    "adv_train_fgsm": "Adv. Train (FGSM)",
    "adv_train_bim": "Adv. Train (BIM)",
    "adv_finetune_fgsm": "Adv. Finetune (FGSM)",
    "adv_finetune_bim": "Adv. Finetune (BIM)",
}

MODE_COLORS = {
    "normal": "#4C72B0",
    "adv_train_fgsm": "#DD8452",
    "adv_train_bim": "#55A868",
    "adv_finetune_fgsm": "#C44E52",
    "adv_finetune_bim": "#8172B2",
}

EPSILON_LABELS = {"eps_0.01": "ε=0.01", "eps_0.03": "ε=0.03", "eps_0.1": "ε=0.1"}
EPSILONS = [0.01, 0.03, 0.1]

st.set_page_config(
    page_title="Audio Classification – CNN Robustness Dashboard",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

@st.cache_data
def load_cv_summary(model: str, mode: str):
    """Load the cv_summary.json for a model/mode. Returns None if missing."""
    pattern = os.path.join(RESULTS_DIR, model, mode, "**", "cv_summary.json")
    hits = glob.glob(pattern, recursive=True)
    # also check direct path (no timestamp subfolder)
    direct = os.path.join(RESULTS_DIR, model, mode, "cv_summary.json")
    if os.path.exists(direct):
        hits.append(direct)
    if not hits:
        return None
    with open(hits[0]) as f:
        return json.load(f)


@st.cache_data
def load_fold_results(model: str, mode: str):
    """Load all fold_N_results.json for a model/mode."""
    pattern_ts = os.path.join(RESULTS_DIR, model, mode, "*", "fold_*_results.json")
    pattern_direct = os.path.join(RESULTS_DIR, model, mode, "fold_*_results.json")
    files = glob.glob(pattern_ts) + glob.glob(pattern_direct)
    folds = {}
    for fp in sorted(files):
        with open(fp) as f:
            data = json.load(f)
        folds[data["fold"]] = data
    return folds


@st.cache_data
def load_all_data(model: str):
    summaries = {}
    folds_data = {}
    for mode in MODES:
        summaries[mode] = load_cv_summary(model, mode)
        folds_data[mode] = load_fold_results(model, mode)
    return summaries, folds_data


# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def mean_adv_accuracy(summary, attack: str):
    """Compute mean adversarial accuracy across folds for each epsilon."""
    if summary is None or attack not in summary:
        return {}
    rows = summary[attack]
    eps_keys = list(next(iter(rows.values())).keys())
    result = {}
    for ek in eps_keys:
        vals = [rows[fk][ek] for fk in rows if ek in rows[fk]]
        result[ek] = float(np.mean(vals)) if vals else None
    return result


def get_metric_df(summaries, metric="accuracy"):
    """Build a DataFrame: rows=modes, cols=[mean, std, available]."""
    rows = []
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s and "mean" in s:
            rows.append(
                {
                    "Mode": label,
                    "mode_key": mode,
                    "Mean": round(s["mean"].get(metric, 0) * 100, 2),
                    "Std": round(s["std"].get(metric, 0) * 100, 2),
                    "Status": "✅ Complete",
                }
            )
        else:
            rows.append(
                {"Mode": label, "mode_key": mode, "Mean": None, "Std": None, "Status": "⏳ Incomplete"}
            )
    return pd.DataFrame(rows)


def get_per_fold_df(summaries, metric="accuracy"):
    """Build a per-fold DataFrame for all complete modes."""
    rows = []
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s and "per_fold" in s:
            for pf in s["per_fold"]:
                rows.append(
                    {
                        "Mode": label,
                        "mode_key": mode,
                        "Fold": pf["fold"],
                        "Value": round(pf.get(metric, 0) * 100, 2),
                    }
                )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.title("🎵 Audio Classification – CNN Robustness Dashboard")
st.caption("UrbanSound8K · 10-Fold Cross-Validation · Normal vs Adversarial Training")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    section = st.radio(
        "Section",
        [
            "📊 Overview",
            "⚔️ Normal vs Adversarial",
            "📈 Training Curves",
            "🛡️ Adversarial Robustness",
            "🔍 Per-Fold Analysis",
        ],
    )
    st.divider()
    metric_choice = st.selectbox(
        "Primary Metric",
        ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    st.divider()
    st.info("Only **CNN** results are shown. More models will be added as experiments complete.")

model = "cnn"
summaries, folds_data = load_all_data(model)
metric_df = get_metric_df(summaries, metric_choice)
per_fold_df = get_per_fold_df(summaries, metric_choice)

# ─── Section: Overview ───────────────────────────────────────────────────────
if section == "📊 Overview":
    st.header("Overview – CNN Model")

    # KPI cards
    complete_modes = metric_df[metric_df["Mean"].notna()]
    cols = st.columns(len(complete_modes))
    for col, (_, row) in zip(cols, complete_modes.iterrows()):
        col.metric(
            label=row["Mode"],
            value=f"{row['Mean']:.1f}%",
            delta=f"±{row['Std']:.1f}%",
            help=metric_choice.replace("_", " ").title(),
        )

    st.divider()

    # Bar chart with error bars
    fig = go.Figure()
    for _, row in metric_df.iterrows():
        if row["Mean"] is None:
            continue
        fig.add_trace(
            go.Bar(
                name=row["Mode"],
                x=[row["Mode"]],
                y=[row["Mean"]],
                error_y=dict(type="data", array=[row["Std"]], visible=True),
                marker_color=MODE_COLORS.get(row["mode_key"], "#888"),
                text=[f"{row['Mean']:.1f}%"],
                textposition="outside",
            )
        )
    fig.update_layout(
        title=f"CNN – Mean {metric_choice.replace('_',' ').title()} Across All Folds",
        yaxis_title=f"{metric_choice.replace('_',' ').title()} (%)",
        yaxis=dict(range=[0, 100]),
        showlegend=False,
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Summary Table")
    display_df = metric_df[["Mode", "Mean", "Std", "Status"]].copy()
    display_df.columns = ["Training Mode", f"Mean {metric_choice.title()} (%)", "Std (%)", "Status"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # All metrics table for complete modes
    st.subheader("All Metrics – Complete Experiments")
    all_rows = []
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s and "mean" in s:
            r = {"Mode": label}
            for m in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]:
                r[m.replace("_", " ").title()] = f"{s['mean'].get(m,0)*100:.2f}%"
            all_rows.append(r)
    if all_rows:
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)


# ─── Section: Normal vs Adversarial ─────────────────────────────────────────
elif section == "⚔️ Normal vs Adversarial":
    st.header("Normal vs Adversarial Training – CNN")

    available_modes = {k: v for k, v in MODES.items() if summaries.get(k) and "mean" in summaries[k]}
    if not available_modes:
        st.warning("No complete experiment results found.")
        st.stop()

    # Radar / Spider chart comparing all metrics
    metrics_list = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    metric_labels = [m.replace("_", " ").title() for m in metrics_list]

    fig_radar = go.Figure()
    for mode, label in available_modes.items():
        s = summaries[mode]
        values = [s["mean"].get(m, 0) * 100 for m in metrics_list]
        values_closed = values + [values[0]]
        labels_closed = metric_labels + [metric_labels[0]]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values_closed,
                theta=labels_closed,
                fill="toself",
                name=label,
                line_color=MODE_COLORS.get(mode, "#888"),
                opacity=0.7,
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="Multi-Metric Comparison (Radar)",
        height=480,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Side-by-side grouped bar chart
    st.subheader("Metric Breakdown")
    fig_bar = go.Figure()
    for mode, label in available_modes.items():
        s = summaries[mode]
        y_vals = [round(s["mean"].get(m, 0) * 100, 2) for m in metrics_list]
        err_vals = [round(s["std"].get(m, 0) * 100, 2) for m in metrics_list]
        fig_bar.add_trace(
            go.Bar(
                name=label,
                x=metric_labels,
                y=y_vals,
                error_y=dict(type="data", array=err_vals, visible=True),
                marker_color=MODE_COLORS.get(mode, "#888"),
            )
        )
    fig_bar.update_layout(
        barmode="group",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 100]),
        height=420,
        title="All Metrics – Normal vs Adversarial Techniques",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # Box plot: distribution of per-fold accuracy
    st.subheader(f"Per-Fold {metric_choice.replace('_',' ').title()} Distribution")
    if not per_fold_df.empty:
        fig_box = px.box(
            per_fold_df,
            x="Mode",
            y="Value",
            color="Mode",
            color_discrete_map={v: MODE_COLORS.get(k, "#888") for k, v in MODES.items()},
            points="all",
            title=f"Distribution of {metric_choice.replace('_',' ').title()} Across 10 Folds",
            labels={"Value": f"{metric_choice.replace('_',' ').title()} (%)"},
            height=420,
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ─── Section: Training Curves ────────────────────────────────────────────────
elif section == "📈 Training Curves":
    st.header("Training Curves – CNN")

    available_modes = {k: v for k, v in MODES.items() if folds_data.get(k)}

    col1, col2 = st.columns(2)
    selected_mode = col1.selectbox("Training Mode", list(available_modes.keys()), format_func=lambda k: MODES[k])
    mode_folds = folds_data.get(selected_mode, {})

    if not mode_folds:
        st.warning("No fold results found for this mode.")
        st.stop()

    fold_options = sorted(mode_folds.keys())
    selected_fold = col2.selectbox("Fold", fold_options, format_func=lambda x: f"Fold {x}")

    fold_result = mode_folds[selected_fold]
    history = fold_result.get("training_history", {})
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))

    if not epochs:
        st.warning("No training history in this fold result.")
        st.stop()

    # Loss and Accuracy curves side by side
    fig_curves = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Loss Curve", "Accuracy Curve"),
    )

    # Loss
    fig_curves.add_trace(
        go.Scatter(x=epochs, y=history.get("train_loss", []), name="Train Loss",
                   line=dict(color="#4C72B0", width=2)),
        row=1, col=1,
    )
    fig_curves.add_trace(
        go.Scatter(x=epochs, y=history.get("val_loss", []), name="Val Loss",
                   line=dict(color="#DD8452", width=2, dash="dash")),
        row=1, col=1,
    )

    # Accuracy
    fig_curves.add_trace(
        go.Scatter(x=epochs, y=history.get("train_acc", []), name="Train Acc",
                   line=dict(color="#55A868", width=2)),
        row=1, col=2,
    )
    fig_curves.add_trace(
        go.Scatter(x=epochs, y=history.get("val_acc", []), name="Val Acc",
                   line=dict(color="#C44E52", width=2, dash="dash")),
        row=1, col=2,
    )

    fig_curves.update_xaxes(title_text="Epoch")
    fig_curves.update_yaxes(title_text="Loss", row=1, col=1)
    fig_curves.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
    fig_curves.update_layout(
        title=f"{MODES[selected_mode]} – Fold {selected_fold} Training Curves",
        height=420,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    # Test metrics for this fold
    tm = fold_result.get("test_metrics", {})
    if tm:
        st.subheader(f"Test Metrics – Fold {selected_fold}")
        m_cols = st.columns(len(tm))
        for col, (k, v) in zip(m_cols, tm.items()):
            col.metric(k.replace("_", " ").title(), f"{v*100:.2f}%")

    st.divider()

    # All folds overlay – accuracy
    st.subheader(f"All Folds – Validation Accuracy Overlay ({MODES[selected_mode]})")
    fig_all = go.Figure()
    all_val_accs = []
    for fold_num, fdata in sorted(mode_folds.items()):
        va = fdata.get("training_history", {}).get("val_acc", [])
        if va:
            all_val_accs.append(va)
            ep = list(range(1, len(va) + 1))
            fig_all.add_trace(
                go.Scatter(
                    x=ep, y=va, name=f"Fold {fold_num}",
                    opacity=0.5, line=dict(width=1),
                )
            )

    # Mean line
    if all_val_accs:
        min_len = min(len(v) for v in all_val_accs)
        arr = np.array([v[:min_len] for v in all_val_accs])
        mean_va = arr.mean(axis=0)
        fig_all.add_trace(
            go.Scatter(
                x=list(range(1, min_len + 1)), y=mean_va.tolist(),
                name="Mean", line=dict(color="black", width=3, dash="dot"),
            )
        )

    fig_all.update_layout(
        xaxis_title="Epoch",
        yaxis_title="Validation Accuracy (%)",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_all, use_container_width=True)


# ─── Section: Adversarial Robustness ─────────────────────────────────────────
elif section == "🛡️ Adversarial Robustness":
    st.header("Adversarial Robustness – CNN")
    st.caption(
        "Accuracy (%) of each trained model when inputs are perturbed with FGSM or BIM attacks at varying ε."
    )

    attack_type = st.radio("Attack Type", ["fgsm", "bim"], horizontal=True, format_func=str.upper)

    # Collect data
    lines = {}
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s is None:
            continue
        adv = mean_adv_accuracy(s, attack_type)
        if adv:
            lines[mode] = adv

    # Also collect clean accuracy for each mode
    clean_accs = {}
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s and "mean" in s:
            clean_accs[mode] = s["mean"].get("accuracy", 0) * 100

    if not lines:
        st.info(
            f"No adversarial accuracy data found for the **{attack_type.upper()}** attack. "
            "Run the attack evaluation script to generate results."
        )
    else:
        # Line chart: epsilon vs accuracy
        fig_rob = go.Figure()

        # Add clean accuracy as leftmost point
        for mode, adv_data in lines.items():
            label = MODES[mode]
            color = MODE_COLORS.get(mode, "#888")
            eps_vals = []
            acc_vals = []

            for ek in ["eps_0.01", "eps_0.03", "eps_0.1"]:
                if ek in adv_data and adv_data[ek] is not None:
                    eps_num = float(ek.replace("eps_", ""))
                    eps_vals.append(eps_num)
                    acc_vals.append(adv_data[ek])

            if eps_vals:
                # Add clean accuracy at eps=0
                clean = clean_accs.get(mode)
                if clean:
                    eps_vals = [0.0] + eps_vals
                    acc_vals = [clean] + acc_vals

                fig_rob.add_trace(
                    go.Scatter(
                        x=eps_vals, y=acc_vals,
                        name=label,
                        mode="lines+markers",
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                    )
                )

        fig_rob.update_layout(
            title=f"{attack_type.upper()} Attack – Accuracy vs Epsilon (ε=0 is clean accuracy)",
            xaxis=dict(
                title="Perturbation Epsilon (ε)",
                tickvals=[0, 0.01, 0.03, 0.1],
                ticktext=["0\n(Clean)", "0.01", "0.03", "0.1"],
            ),
            yaxis=dict(title="Accuracy (%)", range=[0, 100]),
            height=460,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_rob, use_container_width=True)

        # Table of adversarial accuracy
        st.subheader(f"Mean Adversarial Accuracy – {attack_type.upper()} Attack")
        table_rows = []
        for mode, adv_data in lines.items():
            row = {"Training Mode": MODES[mode]}
            row["Clean Acc (%)"] = f"{clean_accs.get(mode, 0):.2f}%" if mode in clean_accs else "N/A"
            for ek in ["eps_0.01", "eps_0.03", "eps_0.1"]:
                val = adv_data.get(ek)
                row[EPSILON_LABELS.get(ek, ek)] = f"{val:.2f}%" if val is not None else "N/A"
            table_rows.append(row)
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Per-fold adversarial accuracy heatmap
        st.subheader(f"Per-Fold Adversarial Accuracy – {attack_type.upper()} (ε=0.03)")
        heatmap_rows = []
        for mode, label in MODES.items():
            s = summaries.get(mode)
            if s is None or attack_type not in s:
                continue
            attack_data = s[attack_type]
            row = {"Mode": label}
            for fold_key in sorted(attack_data.keys(), key=lambda x: int(x.split("_")[1])):
                fold_num = fold_key.split("_")[1]
                val = attack_data[fold_key].get("eps_0.03")
                row[f"Fold {fold_num}"] = round(val, 1) if val is not None else None
            heatmap_rows.append(row)

        if heatmap_rows:
            hm_df = pd.DataFrame(heatmap_rows).set_index("Mode")
            fig_hm = px.imshow(
                hm_df,
                text_auto=".1f",
                color_continuous_scale="RdYlGn",
                zmin=0, zmax=100,
                title=f"Adversarial Accuracy (%) – {attack_type.upper()} at ε=0.03 | Per Fold",
                height=250,
                aspect="auto",
            )
            fig_hm.update_coloraxes(colorbar_title="Acc (%)")
            st.plotly_chart(fig_hm, use_container_width=True)


# ─── Section: Per-Fold Analysis ──────────────────────────────────────────────
elif section == "🔍 Per-Fold Analysis":
    st.header("Per-Fold Analysis – CNN")

    if per_fold_df.empty:
        st.warning("No complete experiment results to analyse.")
        st.stop()

    # Line chart: fold-by-fold accuracy for each mode
    fig_fold = go.Figure()
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s is None or "per_fold" not in s:
            continue
        folds = sorted(s["per_fold"], key=lambda x: x["fold"])
        x = [f["fold"] for f in folds]
        y = [round(f.get(metric_choice, 0) * 100, 2) for f in folds]
        fig_fold.add_trace(
            go.Scatter(
                x=x, y=y,
                name=label,
                mode="lines+markers",
                line=dict(color=MODE_COLORS.get(mode, "#888"), width=2),
                marker=dict(size=8),
            )
        )

    fig_fold.update_layout(
        title=f"Per-Fold {metric_choice.replace('_',' ').title()} – All Techniques",
        xaxis=dict(title="Fold", tickmode="linear", dtick=1),
        yaxis=dict(title=f"{metric_choice.replace('_',' ').title()} (%)", range=[0, 100]),
        height=440,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fold, use_container_width=True)

    st.divider()

    # Per-fold detailed table
    st.subheader("Per-Fold Metrics Table")
    table_rows = []
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s is None or "per_fold" not in s:
            continue
        for pf in sorted(s["per_fold"], key=lambda x: x["fold"]):
            table_rows.append(
                {
                    "Mode": label,
                    "Fold": pf["fold"],
                    "Accuracy": f"{pf.get('accuracy',0)*100:.2f}%",
                    "F1 Macro": f"{pf.get('f1_macro',0)*100:.2f}%",
                    "F1 Weighted": f"{pf.get('f1_weighted',0)*100:.2f}%",
                    "Precision Macro": f"{pf.get('precision_macro',0)*100:.2f}%",
                    "Recall Macro": f"{pf.get('recall_macro',0)*100:.2f}%",
                }
            )
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.divider()

    # Heatmap: mode × fold for selected metric
    st.subheader(f"Heatmap – {metric_choice.replace('_',' ').title()} per Fold × Mode")
    hm_data = {}
    for mode, label in MODES.items():
        s = summaries.get(mode)
        if s is None or "per_fold" not in s:
            continue
        hm_data[label] = {
            f"Fold {pf['fold']}": round(pf.get(metric_choice, 0) * 100, 2)
            for pf in s["per_fold"]
        }

    if hm_data:
        hm_df = pd.DataFrame(hm_data).T
        fold_cols = sorted(hm_df.columns, key=lambda x: int(x.split()[1]))
        hm_df = hm_df[fold_cols]
        fig_hm2 = px.imshow(
            hm_df,
            text_auto=".1f",
            color_continuous_scale="Blues",
            zmin=50, zmax=85,
            title=f"{metric_choice.replace('_',' ').title()} (%) – Mode × Fold",
            height=max(220, 60 * len(hm_data)),
            aspect="auto",
        )
        st.plotly_chart(fig_hm2, use_container_width=True)
