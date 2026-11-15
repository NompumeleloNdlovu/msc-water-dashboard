import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st

DATA_DIR = "saved_outputs"
PROB = "predicted_maintenance_probability"
WSA = "Water Service Authority Name"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#b4bcc4"
INK = "#0f2a3d"
INK2 = "#52606d"
GRID = "#e6ebf0"

st.set_page_config(page_title="Water Infrastructure Maintenance", page_icon="💧", layout="wide")

plt.rcParams.update({
    "figure.dpi": 200,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.labelsize": 9,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "text.color": INK2,
    "font.size": 9,
})

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {font-family: 'Inter', sans-serif;}
    [data-testid="stAppViewContainer"] {background: #f5f7fa;}
    [data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 1.6rem; max-width: 1280px;}
    [data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e3e8ee;}
    .brand {display: flex; align-items: center; gap: 10px; margin: 4px 0 18px 0;}
    .brand-mark {width: 34px; height: 34px; border-radius: 10px; background: linear-gradient(135deg, #0b3c5d, #2a78d6);
                 display: flex; align-items: center; justify-content: center; font-size: 18px;}
    .brand-name {font-weight: 700; color: #0f2a3d; line-height: 1.15; font-size: 0.95rem;}
    .brand-sub {font-weight: 500; color: #7b8794; font-size: 0.72rem;}
    .hero {background: linear-gradient(120deg, #0b3c5d 0%, #2a78d6 100%); padding: 22px 28px; border-radius: 16px; margin-bottom: 14px;}
    .hero-eyebrow {color: #cfe3f7; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;}
    .hero-title {color: #ffffff; font-size: 1.75rem; font-weight: 700; margin: 2px 0 0 0; line-height: 1.2;}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #e3e8ee; border-radius: 14px; padding: 16px 18px;
                              box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);}
    [data-testid="stMetricLabel"] p {color: #52606d; font-size: 0.78rem; font-weight: 500;}
    [data-testid="stMetricValue"] {color: #0f2a3d; font-weight: 700;}
    [data-testid="stVerticalBlockBorderWrapper"] {background: #ffffff; border-radius: 14px;}
    h2, h3 {color: #0f2a3d; font-weight: 600;}
    [data-testid="stCaptionContainer"] {color: #52606d;}
    [data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    .stDownloadButton button {border-radius: 10px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)

required_files = [
    "optimisation_input.csv",
    "scored_output.csv",
    "assessed.csv",
    "unscored.csv",
    "rf_feature_importances.csv",
    "rf_classification_report.csv",
    "reactive_vs_optimised_comparison.csv",
    "within_wsa_std.csv",
]
missing_files = [f for f in required_files if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing_files:
    st.error(f"Missing files in '{DATA_DIR}': {', '.join(missing_files)}")
    st.stop()


def show(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def new_fig(w=6, h=3.8, horizontal=False):
    fig, ax = plt.subplots(figsize=(w, h))
    if horizontal:
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    return fig, ax


def thousands(ax, axis="y"):
    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def labels(ax, container, fmt="{:,.0f}"):
    ax.bar_label(container, labels=[fmt.format(v) for v in container.datavalues], padding=3, color=INK2, fontsize=8)


def end_label(ax, x, y, text):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


@st.cache_data
def load_assets():
    cols = ["Id", "Province", "Bulk Status"]
    assessed = pd.read_csv(f"{DATA_DIR}/assessed.csv", usecols=cols)
    unscored = pd.read_csv(f"{DATA_DIR}/unscored.csv", usecols=cols)
    assessed["Assessment"] = "Assessed"
    unscored["Assessment"] = "Not assessed"
    return pd.concat([assessed, unscored], ignore_index=True)


@st.cache_data
def load_ranking():
    opt = pd.read_csv(f"{DATA_DIR}/optimisation_input.csv")
    opt["ratio"] = opt[PROB] / opt["cost"].replace(0, 1e-9)
    opt = opt.sort_values("ratio", ascending=False).reset_index(drop=True)
    opt["cumulative_cost"] = opt["cost"].cumsum()
    opt["cumulative_priority"] = opt[PROB].cumsum()
    assets = load_assets()[["Id", "Province", "Assessment"]]
    return opt.merge(assets, on="Id", how="left")


@st.cache_data
def load_scored():
    return pd.read_csv(f"{DATA_DIR}/scored_output.csv")


@st.cache_data
def selection_csv(n):
    cols = ["Id", "Province", "Assessment", PROB, "cost"]
    return load_ranking().iloc[:n][cols].to_csv(index=False).encode("utf-8")


def select_n(ranking, budget):
    return int(np.searchsorted(ranking["cumulative_cost"].values, budget, side="right"))


def priority_at(ranking, n):
    return float(ranking["cumulative_priority"].iloc[n - 1]) if n > 0 else 0.0


ICONS = {
    "Overview": "🏠",
    "Predicted risk": "📈",
    "Budget simulator": "💰",
    "Budget sensitivity": "🎚️",
    "Model performance": "🎯",
    "Optimised vs reactive": "⚖️",
    "Authority view": "🏛️",
}

st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-mark">💧</div>
      <div><div class="brand-name">Water Infrastructure</div><div class="brand-sub">Maintenance prioritisation</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "View",
    list(ICONS.keys()),
    format_func=lambda p: f"{ICONS[p]}  {p}",
    label_visibility="collapsed",
)

with st.sidebar.expander("About this dashboard"):
    st.write(
        "Scores come from a random forest trained on assessed pipeline assets. "
        "They are model-based priority scores, not calibrated probabilities of failure. "
        "Scores for assets without an assessment are extrapolations. "
        "Cost values are used as recorded and treated as rand, so budget amounts are indicative."
    )

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-eyebrow">National bulk pipelines</div>
      <div class="hero-title">{ICONS[page]}&nbsp; {page}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "Overview":
    st.caption("National bulk pipeline records and how much of them has a condition assessment.")
    assets = load_assets()
    n_assessed = int((assets["Assessment"] == "Assessed").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pipeline assets", f"{len(assets):,}")
    c2.metric("Assessed", f"{n_assessed:,}")
    c3.metric("Not assessed", f"{len(assets) - n_assessed:,}")
    c4.metric("Provinces", assets["Province"].nunique())

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Assessment coverage by province")
            counts = assets.groupby(["Province", "Assessment"]).size().unstack(fill_value=0)
            for col in ["Assessed", "Not assessed"]:
                if col not in counts.columns:
                    counts[col] = 0
            counts = counts.assign(total=counts.sum(axis=1)).sort_values("total")
            fig, ax = new_fig(horizontal=True)
            ax.barh(counts.index, counts["Assessed"], height=0.62, color=BLUE, edgecolor="white", linewidth=1.5, label="Assessed")
            ax.barh(counts.index, counts["Not assessed"], left=counts["Assessed"], height=0.62, color=GREY, edgecolor="white", linewidth=1.5, label="Not assessed")
            ax.set_xlabel("Number of assets")
            thousands(ax, "x")
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
            show(fig)

    with right:
        with st.container(border=True):
            st.subheader("Bulk status")
            status = assets["Bulk Status"].value_counts()
            fig, ax = new_fig()
            bars = ax.bar(status.index.astype(str), status.values, width=0.55, color=BLUE)
            labels(ax, bars)
            ax.set_ylabel("Number of assets")
            thousands(ax)
            show(fig)

elif page == "Predicted risk":
    st.caption("Model scores for assets that have no condition assessment.")
    scored = load_scored()
    importances = pd.read_csv(f"{DATA_DIR}/rf_feature_importances.csv", index_col=0).iloc[:, 0]

    threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
    c1, c2, c3 = st.columns(3)
    c1.metric("Assets scored", f"{len(scored):,}")
    c2.metric("Mean score", f"{scored[PROB].mean():.3f}")
    c3.metric(f"Assets scoring at least {threshold:.2f}", f"{(scored[PROB] >= threshold).mean() * 100:.1f}%")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Distribution of scores")
            fig, ax = new_fig()
            ax.hist(scored[PROB], bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
            ax.axvline(threshold, color=INK, linewidth=1.4, linestyle="--")
            ax.text(threshold, ax.get_ylim()[1] * 0.97, f" threshold {threshold:.2f}", color=INK2, fontsize=8, va="top", bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85))
            ax.set_xlabel("Predicted maintenance score")
            ax.set_ylabel("Number of assets")
            thousands(ax)
            show(fig)

    with right:
        with st.container(border=True):
            st.subheader("Top 15 feature importances")
            top = importances.head(15).sort_values()
            fig, ax = new_fig(h=4.4, horizontal=True)
            bars = ax.barh(top.index, top.values, height=0.62, color=BLUE)
            labels(ax, bars, "{:.3f}")
            ax.set_xlabel("Impurity-based importance")
            ax.set_xlim(0, top.values.max() * 1.15)
            show(fig)
            st.caption("Importance shows which variables the model relies on, not what causes maintenance need.")

    st.subheader("Highest-scoring assets")
    table = scored
    if WSA in scored.columns:
        authorities = ["All authorities"] + sorted(scored[WSA].dropna().astype(str).unique())
        choice = st.selectbox("Water service authority", authorities)
        if choice != "All authorities":
            table = scored[scored[WSA].astype(str) == choice]
    n_rows = st.slider("Number of assets to display", 10, 200, 25)
    st.dataframe(
        table.sort_values(PROB, ascending=False).head(n_rows),
        hide_index=True,
        column_config={PROB: st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f")},
    )

elif page == "Budget simulator":
    st.caption("Choose a budget and see which assets the ratio rule selects: score divided by cost, highest first.")
    ranking = load_ranking()
    total_cost = ranking["cost"].sum()

    budget_pct = st.slider("Maintenance budget (% of total estimated cost)", 1, 50, 10)
    budget = total_cost * budget_pct / 100
    n = select_n(ranking, budget)
    selected = ranking.iloc[:n]
    n_zero = int((selected["cost"] == 0).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Budget (indicative)", f"R{budget:,.0f}")
    c2.metric("Assets selected", f"{n:,}")
    c3.metric("Budget used", f"{selected['cost'].sum() / budget * 100:.1f}%")
    c4.metric("Priority captured", f"{priority_at(ranking, n):,.1f}")
    c5.metric("Share of total priority", f"{priority_at(ranking, n) / ranking[PROB].sum() * 100:.1f}%")

    if n_zero > 0:
        st.info(
            f"{n_zero:,} of the selected assets have a recorded cost of zero. "
            "They rank first and use none of the budget."
        )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Selected assets by province")
            by_province = selected["Province"].value_counts().sort_values()
            fig, ax = new_fig(horizontal=True)
            bars = ax.barh(by_province.index.astype(str), by_province.values, height=0.62, color=BLUE)
            labels(ax, bars)
            ax.set_xlabel("Assets selected")
            ax.set_xlim(0, max(by_province.values.max(), 1) * 1.15)
            thousands(ax, "x")
            show(fig)

    with right:
        with st.container(border=True):
            st.subheader("Assessed and not assessed")
            by_assessment = selected["Assessment"].value_counts()
            fig, ax = new_fig()
            bars = ax.bar(by_assessment.index.astype(str), by_assessment.values, width=0.5, color=[BLUE, GREY][: len(by_assessment)])
            labels(ax, bars)
            ax.set_ylabel("Assets selected")
            thousands(ax)
            show(fig)

    st.subheader("Recommended maintenance schedule")
    st.caption(f"First 100 of {n:,} selected assets, in selection order.")
    st.dataframe(
        selected[["Id", "Province", "Assessment", PROB, "cost"]].head(100),
        hide_index=True,
        column_config={PROB: st.column_config.ProgressColumn("Score", min_value=0.0, max_value=1.0, format="%.3f")},
    )
    st.download_button(
        "Download all selected assets (CSV)",
        data=selection_csv(n),
        file_name=f"selected_assets_{budget_pct}pct.csv",
        mime="text/csv",
    )

elif page == "Budget sensitivity":
    st.caption("How the total priority captured changes as the budget grows.")
    ranking = load_ranking()
    total_cost = ranking["cost"].sum()

    rows = []
    for pct in [1, 5, 10, 15, 20, 30, 50]:
        budget = total_cost * pct / 100
        n = select_n(ranking, budget)
        used = float(ranking["cumulative_cost"].iloc[n - 1]) if n > 0 else 0.0
        rows.append({
            "Budget (%)": pct,
            "Budget (R)": budget,
            "Assets selected": n,
            "Budget used (%)": used / budget * 100,
            "Total priority captured": priority_at(ranking, n),
        })
    sens = pd.DataFrame(rows)
    sens["Priority gained per extra 1%"] = sens["Total priority captured"].diff() / sens["Budget (%)"].diff()
    sens.loc[0, "Priority gained per extra 1%"] = sens.loc[0, "Total priority captured"] / sens.loc[0, "Budget (%)"]

    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            st.subheader("Priority captured")
            fig, ax = new_fig(5, 3.4)
            ax.plot(sens["Budget (%)"], sens["Total priority captured"], marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, linewidth=2, color=BLUE)
            ax.set_xlabel("Budget (% of total cost)")
            ax.set_ylabel("Total priority captured")
            thousands(ax)
            show(fig)

    with c2:
        with st.container(border=True):
            st.subheader("Assets selected")
            fig, ax = new_fig(5, 3.4)
            ax.plot(sens["Budget (%)"], sens["Assets selected"], marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5, linewidth=2, color=BLUE)
            ax.set_xlabel("Budget (% of total cost)")
            ax.set_ylabel("Assets selected")
            thousands(ax)
            show(fig)

    with c3:
        with st.container(border=True):
            st.subheader("Gain per extra 1% of budget")
            fig, ax = new_fig(5, 3.4)
            bars = ax.bar(sens["Budget (%)"].astype(str), sens["Priority gained per extra 1%"], width=0.6, color=BLUE)
            labels(ax, bars, "{:,.0f}")
            ax.set_xlabel("Budget level (%)")
            ax.set_ylabel("Priority per percentage point")
            thousands(ax)
            show(fig)

    st.dataframe(
        sens.round(2),
        hide_index=True,
    )

elif page == "Model performance":
    st.caption("Random forest performance on the held-out test set (20% of assessed assets, one random split).")
    report = pd.read_csv(f"{DATA_DIR}/rf_classification_report.csv", index_col=0)
    support0 = report.loc["0", "support"]
    support1 = report.loc["1", "support"]
    base_rate = support1 / (support0 + support1)
    base_f1 = 2 * base_rate / (1 + base_rate)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{report.loc['accuracy', 'precision']:.3f}")
    c2.metric("F1, needs maintenance", f"{report.loc['1', 'f1-score']:.3f}")
    c3.metric("F1, no maintenance needed", f"{report.loc['0', 'f1-score']:.3f}")
    c4.metric("Predict-all-positive accuracy", f"{base_rate:.3f}")

    st.caption(
        f"A classifier that labels every asset as needing maintenance would score {base_rate:.3f} accuracy "
        f"and {base_f1:.3f} F1 for the positive class, so the no-maintenance class is the more informative check."
    )

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Precision, recall and F1 by class")
            metrics = ["precision", "recall", "f1-score"]
            x = np.arange(len(metrics))
            fig, ax = new_fig()
            b0 = ax.bar(x - 0.2, [report.loc["0", m] for m in metrics], width=0.36, color=GREY, edgecolor="white", linewidth=1.5, label="No maintenance needed")
            b1 = ax.bar(x + 0.2, [report.loc["1", m] for m in metrics], width=0.36, color=BLUE, edgecolor="white", linewidth=1.5, label="Needs maintenance")
            labels(ax, b0, "{:.3f}")
            labels(ax, b1, "{:.3f}")
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1.12)
            ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
            show(fig)

    with right:
        with st.container(border=True):
            st.subheader("Full report")
            st.dataframe(report.round(3))

elif page == "Optimised vs reactive":
    st.caption(
        "Test assets only. The reactive strategy is a simulated random order, averaged over repeated runs, "
        "not an observed maintenance practice. Capture is measured against inspection-based labels."
    )
    comparison = pd.read_csv(f"{DATA_DIR}/reactive_vs_optimised_comparison.csv")
    comparison["Selected that are genuine needs"] = comparison["true_positives_captured"] / comparison["n_selected"]
    colours = {"ML-prioritised": BLUE, "Reactive (random)": ORANGE}

    levels = sorted(comparison["budget"].unique())
    level = st.select_slider("Budget level", options=levels, format_func=lambda v: f"R{v / 1e6:,.1f} million")
    at_level = comparison[comparison["budget"] == level].set_index("strategy")

    if "ML-prioritised" in at_level.index and "Reactive (random)" in at_level.index:
        ml = at_level.loc["ML-prioritised"]
        rx = at_level.loc["Reactive (random)"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Share of needs captured, ML", f"{ml['recall_of_true_needs'] * 100:.1f}%")
        c2.metric("Share of needs captured, reactive", f"{rx['recall_of_true_needs'] * 100:.1f}%")
        c3.metric("Difference", f"{(ml['recall_of_true_needs'] - rx['recall_of_true_needs']) * 100:+.1f} points")

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.subheader("Share of genuine needs captured")
            fig, ax = new_fig()
            for strategy, group in comparison.groupby("strategy"):
                ax.plot(group["budget"] / 1e6, group["recall_of_true_needs"], marker="o", markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5, linewidth=2,
                        color=colours.get(strategy, GREY), label=strategy)
            ax.axvline(level / 1e6, color=INK, linewidth=1.2, linestyle="--")
            ax.set_xlabel("Budget (R million)")
            ax.set_ylabel("Share of genuine needs captured")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
            show(fig)

    with right:
        with st.container(border=True):
            st.subheader("Assets selected")
            fig, ax = new_fig()
            for strategy, group in comparison.groupby("strategy"):
                ax.plot(group["budget"] / 1e6, group["n_selected"], marker="o", markersize=6,
                        markeredgecolor="white", markeredgewidth=1.5, linewidth=2,
                        color=colours.get(strategy, GREY), label=strategy)
            ax.axvline(level / 1e6, color=INK, linewidth=1.2, linestyle="--")
            ax.set_xlabel("Budget (R million)")
            ax.set_ylabel("Assets selected")
            thousands(ax)
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
            show(fig)

    st.info(
        "Many test assets have a recorded cost of zero. The ratio rule selects them first, "
        "so the ML strategy can reach a large share of needs at small budgets."
    )
    table = comparison[["strategy", "budget", "n_selected", "true_positives_captured",
                        "recall_of_true_needs", "Selected that are genuine needs"]].copy()
    table["budget"] = table["budget"] / 1e6
    st.dataframe(table.rename(columns={"budget": "budget (R million)"}).round(3), hide_index=True)

elif page == "Authority view":
    st.caption("Whether scores separate authorities from one another or assets within the same authority.")
    wsa = pd.read_csv(f"{DATA_DIR}/within_wsa_std.csv", index_col=0)
    scored = load_scored()

    avg_within = float(wsa["std"].mean())
    overall = float(scored[PROB].std())

    c1, c2, c3 = st.columns(3)
    c1.metric("Average within-authority std", f"{avg_within:.4f}")
    c2.metric("Std across all scored assets", f"{overall:.4f}")
    c3.metric("Ratio", f"{avg_within / overall:.2f}")

    if avg_within / overall < 0.5:
        st.info("Scores vary much less within an authority than across all assets, so they separate authorities more than individual assets.")
    else:
        st.info("Scores vary within authorities about as much as across all assets, so they also separate individual assets.")

    top_n = st.slider("Number of authorities to show", 5, 30, 15)
    top_wsa = wsa.sort_values("count", ascending=False).head(top_n)

    with st.container(border=True):
        st.subheader("Mean score by authority")
        fig, ax = new_fig(8, max(3.6, 0.32 * top_n + 1), horizontal=True)
        plot_data = top_wsa["mean"].sort_values()
        bars = ax.barh(plot_data.index.astype(str), plot_data.values, height=0.62, color=BLUE)
        labels(ax, bars, "{:.2f}")
        ax.set_xlabel("Mean predicted maintenance score")
        ax.set_xlim(0, max(plot_data.values.max(), 0.01) * 1.12)
        show(fig)

    st.dataframe(
        top_wsa.rename(columns={"mean": "mean score", "std": "within-authority std", "count": "assets scored"}).round(4)
    )

st.caption("Decision-support prototype only. It does not replace engineering judgement, inspections or municipal planning.")
