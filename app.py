import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

DATA_DIR = "saved_outputs"
PROB = "predicted_maintenance_probability"
WSA = "Water Service Authority Name"

BLUE = "#0B6E99"
TEAL = "#2A9D8F"
AMBER = "#E9A23B"
GREY = "#8A9BA5"
RED = "#C8553D"

st.set_page_config(page_title="Water Infrastructure Maintenance Dashboard", layout="wide")

plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#E3EAEE",
    "axes.edgecolor": GREY,
    "font.size": 10,
})

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {background-color: #EEF4F7; padding: 12px 16px; border-radius: 8px;}
    h1 {font-size: 2rem;}
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


st.sidebar.title("Water Infrastructure Maintenance")
page = st.sidebar.radio(
    "View",
    [
        "Overview",
        "Predicted risk",
        "Budget simulator",
        "Budget sensitivity",
        "Model performance",
        "Optimised vs reactive",
        "Authority view",
    ],
)

with st.sidebar.expander("About this dashboard"):
    st.write(
        "Scores come from a random forest trained on assessed pipeline assets. "
        "They are model-based priority scores, not calibrated probabilities of failure. "
        "Scores for assets without an assessment are extrapolations. "
        "Cost values are used as recorded and treated as rand, so budget amounts are indicative."
    )

st.title(page)

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
        st.subheader("Assessment coverage by province")
        counts = assets.groupby(["Province", "Assessment"]).size().unstack(fill_value=0)
        for col in ["Assessed", "Not assessed"]:
            if col not in counts.columns:
                counts[col] = 0
        counts = counts.assign(total=counts.sum(axis=1)).sort_values("total")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(counts.index, counts["Assessed"], color=BLUE, label="Assessed")
        ax.barh(counts.index, counts["Not assessed"], left=counts["Assessed"], color=GREY, label="Not assessed")
        ax.set_xlabel("Number of assets")
        ax.legend(frameon=False)
        show(fig)

    with right:
        st.subheader("Bulk status")
        status = assets["Bulk Status"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(status.index.astype(str), status.values, color=TEAL)
        ax.set_ylabel("Number of assets")
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
        st.subheader("Distribution of scores")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(scored[PROB], bins=40, color=BLUE)
        ax.axvline(threshold, color=AMBER, linewidth=2)
        ax.set_xlabel("Predicted maintenance score")
        ax.set_ylabel("Number of assets")
        show(fig)

    with right:
        st.subheader("Top 15 feature importances")
        top = importances.head(15).sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values, color=TEAL)
        ax.set_xlabel("Impurity-based importance")
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
        st.subheader("Selected assets by province")
        by_province = selected["Province"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(by_province.index.astype(str), by_province.values, color=BLUE)
        ax.set_xlabel("Assets selected")
        show(fig)

    with right:
        st.subheader("Assessed and not assessed")
        by_assessment = selected["Assessment"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(by_assessment.index.astype(str), by_assessment.values, color=[BLUE, GREY][: len(by_assessment)])
        ax.set_ylabel("Assets selected")
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
        st.subheader("Priority captured")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(sens["Budget (%)"], sens["Total priority captured"], marker="o", color=BLUE)
        ax.set_xlabel("Budget (% of total cost)")
        ax.set_ylabel("Total priority captured")
        show(fig)

    with c2:
        st.subheader("Assets selected")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(sens["Budget (%)"], sens["Assets selected"], marker="o", color=AMBER)
        ax.set_xlabel("Budget (% of total cost)")
        ax.set_ylabel("Assets selected")
        show(fig)

    with c3:
        st.subheader("Gain per extra 1% of budget")
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(sens["Budget (%)"].astype(str), sens["Priority gained per extra 1%"], color=TEAL)
        ax.set_xlabel("Budget level (%)")
        ax.set_ylabel("Priority per percentage point")
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
        st.subheader("Precision, recall and F1 by class")
        metrics = ["precision", "recall", "f1-score"]
        x = np.arange(len(metrics))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x - 0.2, [report.loc["0", m] for m in metrics], width=0.4, color=GREY, label="No maintenance needed")
        ax.bar(x + 0.2, [report.loc["1", m] for m in metrics], width=0.4, color=BLUE, label="Needs maintenance")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, loc="lower right")
        show(fig)

    with right:
        st.subheader("Full report")
        st.dataframe(report.round(3))

elif page == "Optimised vs reactive":
    st.caption(
        "Test assets only. The reactive strategy is a simulated random order, averaged over repeated runs, "
        "not an observed maintenance practice. Capture is measured against inspection-based labels."
    )
    comparison = pd.read_csv(f"{DATA_DIR}/reactive_vs_optimised_comparison.csv")
    comparison["Selected that are genuine needs"] = comparison["true_positives_captured"] / comparison["n_selected"]
    colours = {"ML-prioritised": BLUE, "Reactive (random)": RED}

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
        st.subheader("Share of genuine needs captured")
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, group in comparison.groupby("strategy"):
            ax.plot(group["budget"] / 1e6, group["recall_of_true_needs"], marker="o",
                    color=colours.get(strategy, GREY), label=strategy)
        ax.axvline(level / 1e6, color=AMBER, linewidth=2)
        ax.set_xlabel("Budget (R million)")
        ax.set_ylabel("Share of genuine needs captured")
        ax.legend(frameon=False)
        show(fig)

    with right:
        st.subheader("Assets selected")
        fig, ax = plt.subplots(figsize=(6, 4))
        for strategy, group in comparison.groupby("strategy"):
            ax.plot(group["budget"] / 1e6, group["n_selected"], marker="o",
                    color=colours.get(strategy, GREY), label=strategy)
        ax.axvline(level / 1e6, color=AMBER, linewidth=2)
        ax.set_xlabel("Budget (R million)")
        ax.set_ylabel("Assets selected")
        ax.legend(frameon=False)
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

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_data = top_wsa["mean"].sort_values()
    ax.barh(plot_data.index.astype(str), plot_data.values, color=BLUE)
    ax.set_xlabel("Mean predicted maintenance score")
    show(fig)

    st.dataframe(
        top_wsa.rename(columns={"mean": "mean score", "std": "within-authority std", "count": "assets scored"}).round(4)
    )

st.caption("Decision-support prototype only. It does not replace engineering judgement, inspections or municipal planning.")
