import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="LOS Attribution Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SPECIALTY_CONFIG = {
    "GenMed1": {"mean_los": 4.4, "sd_los": 1.0, "weight": 0.22},
    "GenMed2": {"mean_los": 5.1, "sd_los": 1.2, "weight": 0.24},
    "Geriatrics": {"mean_los": 6.0, "sd_los": 1.4, "weight": 0.20},
    "Neurology": {"mean_los": 5.8, "sd_los": 1.4, "weight": 0.16},
    "Respiratory": {"mean_los": 5.3, "sd_los": 1.3, "weight": 0.18},
}

SPECIALTIES = list(SPECIALTY_CONFIG.keys())
SPECIALTY_WEIGHTS = [SPECIALTY_CONFIG[s]["weight"] for s in SPECIALTIES]

SCENARIOS = {
    "Custom": {
        "global_diag": 0,
        "global_allied": 0,
        "global_dest": 0,
        "global_weekend": 0,
        "neuro_resp_diag": 0,
        "geri_dest": 0,
        "geri_allied": 0,
        "weekend_discharge": 0,
    },
    "Faster Diagnostics": {
        "global_diag": 25,
        "global_allied": 0,
        "global_dest": 0,
        "global_weekend": 0,
        "neuro_resp_diag": 20,
        "geri_dest": 0,
        "geri_allied": 0,
        "weekend_discharge": 0,
    },
    "Better Destination Flow": {
        "global_diag": 0,
        "global_allied": 0,
        "global_dest": 25,
        "global_weekend": 0,
        "neuro_resp_diag": 0,
        "geri_dest": 20,
        "geri_allied": 0,
        "weekend_discharge": 0,
    },
    "Weekend Improvement": {
        "global_diag": 0,
        "global_allied": 0,
        "global_dest": 0,
        "global_weekend": 20,
        "neuro_resp_diag": 0,
        "geri_dest": 0,
        "geri_allied": 20,
        "weekend_discharge": 25,
    },
    "Combined Package": {
        "global_diag": 20,
        "global_allied": 15,
        "global_dest": 20,
        "global_weekend": 15,
        "neuro_resp_diag": 15,
        "geri_dest": 15,
        "geri_allied": 20,
        "weekend_discharge": 20,
    },
}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def assess_data_quality(df: pd.DataFrame):
    required_columns = [
        "episode_id",
        "specialty",
        "admit_datetime",
        "discharge_datetime",
    ]

    optional_columns = [
        "age",
        "diagnosis_group",
        "destination_type",
        "medically_ready_datetime",
        "diagnostics_delay_days",
        "allied_delay_days",
        "destination_delay_days",
        "discharge_process_delay_days",
    ]

    results = {
        "missing_required": [],
        "missing_optional": [],
        "invalid_admit_datetime": 0,
        "invalid_discharge_datetime": 0,
        "negative_or_zero_los": 0,
        "missing_specialty": 0,
        "missing_episode_id": 0,
        "optional_completeness": {},
        "score": 0,
    }

    for col in required_columns:
        if col not in df.columns:
            results["missing_required"].append(col)

    for col in optional_columns:
        if col not in df.columns:
            results["missing_optional"].append(col)

    if results["missing_required"]:
        results["score"] = max(0, 100 - 25 * len(results["missing_required"]))
        return results

    results["missing_specialty"] = int(df["specialty"].isna().sum())
    results["missing_episode_id"] = int(df["episode_id"].isna().sum())

    admit_parsed = pd.to_datetime(df["admit_datetime"], errors="coerce")
    discharge_parsed = pd.to_datetime(df["discharge_datetime"], errors="coerce")

    results["invalid_admit_datetime"] = int(admit_parsed.isna().sum())
    results["invalid_discharge_datetime"] = int(discharge_parsed.isna().sum())

    los_days = (discharge_parsed - admit_parsed).dt.total_seconds() / (3600 * 24)
    results["negative_or_zero_los"] = int((los_days <= 0).fillna(False).sum())

    for col in optional_columns:
        if col in df.columns:
            completeness = 100 * (1 - df[col].isna().mean())
            results["optional_completeness"][col] = round(completeness, 1)

    score = 100
    score -= 25 * len(results["missing_required"])
    score -= min(15, results["missing_specialty"] * 2)
    score -= min(15, results["missing_episode_id"] * 2)
    score -= min(20, results["invalid_admit_datetime"] * 2)
    score -= min(20, results["invalid_discharge_datetime"] * 2)
    score -= min(20, results["negative_or_zero_los"] * 2)

    if results["optional_completeness"]:
        avg_optional = np.mean(list(results["optional_completeness"].values()))
        if avg_optional >= 90:
            pass
        elif avg_optional >= 75:
            score -= 5
        elif avg_optional >= 50:
            score -= 10
        else:
            score -= 15

    results["score"] = int(max(0, min(100, score)))
    return results


def generate_data(n=300, seed=42):
    np.random.seed(seed)
    rows = []

    for i in range(n):
        specialty = np.random.choice(SPECIALTIES, p=SPECIALTY_WEIGHTS)
        cfg = SPECIALTY_CONFIG[specialty]

        expected = max(1.5, np.random.normal(cfg["mean_los"], cfg["sd_los"]))

        if specialty == "Neurology":
            diagnostics = np.random.exponential(1.1)
            allied = np.random.exponential(0.7)
            destination = np.random.exponential(1.3)
            weekend = np.random.exponential(0.45)
        elif specialty == "Respiratory":
            diagnostics = np.random.exponential(0.9)
            allied = np.random.exponential(0.6)
            destination = np.random.exponential(1.1)
            weekend = np.random.exponential(0.5)
        elif specialty == "Geriatrics":
            diagnostics = np.random.exponential(0.8)
            allied = np.random.exponential(0.9)
            destination = np.random.exponential(1.6)
            weekend = np.random.exponential(0.5)
        elif specialty == "GenMed2":
            diagnostics = np.random.exponential(0.85)
            allied = np.random.exponential(0.7)
            destination = np.random.exponential(1.25)
            weekend = np.random.exponential(0.4)
        else:
            diagnostics = np.random.exponential(0.7)
            allied = np.random.exponential(0.5)
            destination = np.random.exponential(1.0)
            weekend = np.random.exponential(0.35)

        discharge = np.random.exponential(0.3)
        actual = expected + diagnostics + allied + destination + weekend + discharge

        rows.append(
            {
                "patient_id": f"P{i+1:04d}",
                "episode_id": f"E{i+1:04d}",
                "specialty": specialty,
                "expected_los": expected,
                "diagnostics": diagnostics,
                "allied": allied,
                "destination": destination,
                "weekend": weekend,
                "discharge": discharge,
                "actual_los": actual,
            }
        )

    return pd.DataFrame(rows)


def infer_delays_from_timestamps(df: pd.DataFrame):
    out = df.copy()

    if "medically_ready_datetime" not in out.columns:
        out["clinical_los"] = out["actual_los"]
        out["non_clinical_delay"] = 0.0
        return out

    mrd = pd.to_datetime(out["medically_ready_datetime"], errors="coerce")

    out["clinical_los"] = (
        (mrd - out["admit_datetime"]).dt.total_seconds() / (3600 * 24)
    )
    out["non_clinical_delay"] = (
        (out["discharge_datetime"] - mrd).dt.total_seconds() / (3600 * 24)
    )

    out["clinical_los"] = out["clinical_los"].clip(lower=0).fillna(0)
    out["non_clinical_delay"] = out["non_clinical_delay"].clip(lower=0).fillna(0)

    return out


def apply_interventions(
    df: pd.DataFrame,
    global_diag_reduction: int,
    global_allied_reduction: int,
    global_dest_reduction: int,
    global_weekend_reduction: int,
    neuro_resp_diag_bonus: int,
    geri_dest_bonus: int,
    geri_allied_bonus: int,
    weekend_discharge_bonus: int,
):
    out = df.copy()

    for col in ["diagnostics", "allied", "destination", "weekend", "discharge"]:
        if col not in out.columns:
            out[col] = 0.0

    out["diagnostics"] *= (1 - global_diag_reduction / 100)
    out["allied"] *= (1 - global_allied_reduction / 100)
    out["destination"] *= (1 - global_dest_reduction / 100)
    out["weekend"] *= (1 - global_weekend_reduction / 100)

    neuro_resp_mask = out["specialty"].isin(["Neurology", "Respiratory"])
    out.loc[neuro_resp_mask, "diagnostics"] *= (1 - neuro_resp_diag_bonus / 100)

    geri_mask = out["specialty"] == "Geriatrics"
    out.loc[geri_mask, "destination"] *= (1 - geri_dest_bonus / 100)
    out.loc[geri_mask, "allied"] *= (1 - geri_allied_bonus / 100)

    out["weekend"] *= (1 - weekend_discharge_bonus / 100)

    out["actual_los"] = (
        out["expected_los"]
        + out["diagnostics"]
        + out["allied"]
        + out["destination"]
        + out["weekend"]
        + out["discharge"]
    )

    return out


def scenario_metrics_table(baseline_df, scenario_df, cost_per_bed_day):
    baseline_mean = baseline_df["actual_los"].mean()
    scenario_mean = scenario_df["actual_los"].mean()
    los_reduction = baseline_mean - scenario_mean
    bed_days_saved = baseline_df["actual_los"].sum() - scenario_df["actual_los"].sum()
    est_savings = bed_days_saved * cost_per_bed_day

    return {
        "Mean LOS": scenario_mean,
        "LOS Reduction": los_reduction,
        "Bed-Days": scenario_df["actual_los"].sum(),
        "Bed-Days Saved": bed_days_saved,
        "Estimated Savings": est_savings,
    }


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("LOS Simulator")

exec_mode = st.sidebar.checkbox("Executive mode", value=True)
scenario_mode = st.sidebar.checkbox("Scenario comparison mode", value=True)

scenario = st.sidebar.selectbox("Select scenario", list(SCENARIOS.keys()))
scenario_name = st.sidebar.text_input("Name this scenario", value=scenario)
preset = SCENARIOS[scenario]

st.sidebar.header("Population")
n_patients = st.sidebar.slider("Number of simulated patients", 50, 1000, 300)

selected_specialties = st.sidebar.multiselect(
    "Specialties included",
    SPECIALTIES,
    default=SPECIALTIES,
)

uploaded_file = st.sidebar.file_uploader("Upload real dataset (CSV)", type=["csv"])

st.sidebar.header("Financial Assumptions")
cost_per_bed_day = st.sidebar.number_input(
    "Estimated cost per bed-day ($)",
    min_value=500,
    max_value=5000,
    value=1500,
    step=100
)

st.sidebar.header("General Interventions")
global_diag_reduction = st.sidebar.slider(
    "Reduce diagnostic delay across all specialties (%)",
    0, 100, preset["global_diag"]
)
global_allied_reduction = st.sidebar.slider(
    "Reduce allied health delay across all specialties (%)",
    0, 100, preset["global_allied"]
)
global_dest_reduction = st.sidebar.slider(
    "Reduce destination delay across all specialties (%)",
    0, 100, preset["global_dest"]
)
global_weekend_reduction = st.sidebar.slider(
    "Reduce weekend-related delay across all specialties (%)",
    0, 100, preset["global_weekend"]
)

st.sidebar.header("Targeted Interventions")
neuro_resp_diag_bonus = st.sidebar.slider(
    "Further reduce diagnostic delay for Neurology and Respiratory patients (%)",
    0, 100, preset["neuro_resp_diag"]
)
geri_dest_bonus = st.sidebar.slider(
    "Further reduce destination delay for Geriatrics patients (%)",
    0, 100, preset["geri_dest"]
)
geri_allied_bonus = st.sidebar.slider(
    "Further reduce allied health delay for Geriatrics patients (%)",
    0, 100, preset["geri_allied"]
)
weekend_discharge_bonus = st.sidebar.slider(
    "Improve weekend discharge processes and reduce weekend delay (%)",
    0, 100, preset["weekend_discharge"]
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
quality = None
preview_df = None

if uploaded_file is not None:
    preview_df = pd.read_csv(uploaded_file)
    quality = assess_data_quality(preview_df)

    st.subheader("Data Quality Assessment")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Data quality score", f"{quality['score']}/100")
    q2.metric("Missing required columns", len(quality["missing_required"]))
    q3.metric(
        "Invalid timestamps",
        quality["invalid_admit_datetime"] + quality["invalid_discharge_datetime"]
    )
    q4.metric("Invalid LOS rows", quality["negative_or_zero_los"])

    if quality["score"] >= 85:
        st.success("High-quality dataset: suitable for real modelling.")
    elif quality["score"] >= 65:
        st.warning("Moderate-quality dataset: usable, but some fields need improvement.")
    else:
        st.error("Low-quality dataset: fix key data issues before using for decision-making.")

    st.subheader("Uploaded Data Preview")
    st.dataframe(preview_df.head(), width="stretch")

    optional_df = pd.DataFrame(
        {
            "Column": list(quality["optional_completeness"].keys()),
            "Completeness (%)": list(quality["optional_completeness"].values()),
        }
    )
    if not optional_df.empty:
        st.markdown("### Optional Field Completeness")
        st.dataframe(optional_df, width="stretch")

    with st.expander("Show uploaded dataset structure"):
        st.write("Columns detected:")
        st.write(list(preview_df.columns))
        st.write("Dataset shape:")
        st.write(preview_df.shape)

    if quality["missing_required"]:
        st.error(f"Missing required columns: {quality['missing_required']}")
        st.stop()

    if quality["invalid_admit_datetime"] > 0 or quality["invalid_discharge_datetime"] > 0:
        st.error("Uploaded dataset contains invalid datetime values. Please correct them before modelling.")
        st.stop()

    if quality["negative_or_zero_los"] > 0:
        st.error("Uploaded dataset contains zero or negative LOS rows. Please correct them before modelling.")
        st.stop()

    if quality["missing_episode_id"] > 0:
        st.error("Uploaded dataset contains missing episode_id values. Please correct them before modelling.")
        st.stop()

    if quality["missing_specialty"] > 0:
        st.error("Uploaded dataset contains missing specialty values. Please correct them before modelling.")
        st.stop()

    baseline = preview_df.copy()

    baseline["admit_datetime"] = pd.to_datetime(baseline["admit_datetime"], errors="coerce")
    baseline["discharge_datetime"] = pd.to_datetime(baseline["discharge_datetime"], errors="coerce")

    baseline["actual_los"] = (
        (baseline["discharge_datetime"] - baseline["admit_datetime"]).dt.total_seconds() / (3600 * 24)
    )

    baseline["expected_los"] = baseline.groupby("specialty")["actual_los"].transform("mean")
    baseline = infer_delays_from_timestamps(baseline)

    optional_delay_defaults = {
        "diagnostics_delay_days": 0.0,
        "allied_delay_days": 0.0,
        "destination_delay_days": 0.0,
        "discharge_process_delay_days": 0.0,
    }

    for col, default in optional_delay_defaults.items():
        if col not in baseline.columns:
            baseline[col] = default
        else:
            baseline[col] = pd.to_numeric(baseline[col], errors="coerce").fillna(default)

    if "non_clinical_delay" in baseline.columns:
        baseline["destination"] = baseline["non_clinical_delay"] * 0.6
        baseline["discharge"] = baseline["non_clinical_delay"] * 0.4
    else:
        baseline["destination"] = baseline.get("destination_delay_days", 0.0)
        baseline["discharge"] = baseline.get("discharge_process_delay_days", 0.0)

    baseline["diagnostics"] = baseline.get("diagnostics_delay_days", 0.0)
    baseline["allied"] = baseline.get("allied_delay_days", 0.0)
    baseline["weekend"] = baseline["admit_datetime"].dt.dayofweek.isin([5, 6]).astype(float) * 0.3

    baseline = baseline.dropna(subset=["episode_id", "specialty", "admit_datetime", "discharge_datetime"])
    baseline = baseline[baseline["actual_los"] > 0].copy()

    if baseline.empty:
        st.error("No valid rows remain after validation and cleaning.")
        st.stop()

else:
    baseline = generate_data(n_patients, seed=42)

if "specialty" in baseline.columns and selected_specialties:
    baseline = baseline[baseline["specialty"].isin(selected_specialties)].copy()

if len(baseline) == 0:
    st.warning("No data available after filtering.")
    st.stop()

# --------------------------------------------------
# APPLY MAIN SCENARIO
# --------------------------------------------------
df = apply_interventions(
    baseline,
    global_diag_reduction=global_diag_reduction,
    global_allied_reduction=global_allied_reduction,
    global_dest_reduction=global_dest_reduction,
    global_weekend_reduction=global_weekend_reduction,
    neuro_resp_diag_bonus=neuro_resp_diag_bonus,
    geri_dest_bonus=geri_dest_bonus,
    geri_allied_bonus=geri_allied_bonus,
    weekend_discharge_bonus=weekend_discharge_bonus,
)

# --------------------------------------------------
# MAIN METRICS
# --------------------------------------------------
baseline_los = baseline["actual_los"].mean()
new_los = df["actual_los"].mean()
los_reduction = baseline_los - new_los
bed_days_saved = baseline["actual_los"].sum() - df["actual_los"].sum()
financial_savings = bed_days_saved * cost_per_bed_day

# --------------------------------------------------
# EXEC INSIGHTS
# --------------------------------------------------
delay_means = (
    df[["diagnostics", "allied", "destination", "weekend"]]
    .mean()
    .sort_values(ascending=False)
)

top_driver_code = delay_means.index[0]
driver_label_map = {
    "diagnostics": "diagnostic delay",
    "allied": "allied health delay",
    "destination": "destination delay",
    "weekend": "weekend-related delay",
}
top_driver_label = driver_label_map.get(top_driver_code, top_driver_code)

specialty_mean_los = (
    df.groupby("specialty")["actual_los"]
    .mean()
    .sort_values(ascending=False)
)

top_specialty = specialty_mean_los.index[0]
top_specialty_los = specialty_mean_los.iloc[0]

los_reduction_pct = 0.0
if baseline_los > 0:
    los_reduction_pct = (los_reduction / baseline_los) * 100

exec_headline = (
    f"LOS is highest in {top_specialty}, and the largest overall driver is {top_driver_label}."
)
exec_summary = (
    f"This scenario reduces mean LOS by {los_reduction:.2f} days ({los_reduction_pct:.1f}%), "
    f"saves approximately {bed_days_saved:.0f} bed-days, and is associated with estimated cost savings of "
    f"${financial_savings:,.0f}."
)
exec_action = (
    f"The strongest operational lever is to target {top_driver_label}, particularly in {top_specialty}, "
    f"where mean LOS is {top_specialty_los:.2f} days."
)

# --------------------------------------------------
# SCENARIO ENGINE
# --------------------------------------------------
scenario_table = None
auto_recommendation = None

if scenario_mode:
    scenario_rows = []

    for scenario_label, settings in SCENARIOS.items():
        scenario_df = apply_interventions(
            baseline,
            global_diag_reduction=settings["global_diag"],
            global_allied_reduction=settings["global_allied"],
            global_dest_reduction=settings["global_dest"],
            global_weekend_reduction=settings["global_weekend"],
            neuro_resp_diag_bonus=settings["neuro_resp_diag"],
            geri_dest_bonus=settings["geri_dest"],
            geri_allied_bonus=settings["geri_allied"],
            weekend_discharge_bonus=settings["weekend_discharge"],
        )

        metrics = scenario_metrics_table(baseline, scenario_df, cost_per_bed_day)
        scenario_rows.append(
            {
                "Scenario": scenario_label,
                "Mean LOS": metrics["Mean LOS"],
                "LOS Reduction": metrics["LOS Reduction"],
                "Bed-Days": metrics["Bed-Days"],
                "Bed-Days Saved": metrics["Bed-Days Saved"],
                "Estimated Savings": metrics["Estimated Savings"],
            }
        )

    scenario_table = pd.DataFrame(scenario_rows).sort_values("Mean LOS", ascending=True)

    if not scenario_table.empty:
        ranked = scenario_table.sort_values(
            ["Estimated Savings", "Mean LOS"],
            ascending=[False, True]
        ).reset_index(drop=True)

        best_row = ranked.iloc[0]

        auto_recommendation = {
            "scenario": best_row["Scenario"],
            "mean_los": best_row["Mean LOS"],
            "los_reduction": best_row["LOS Reduction"],
            "bed_days_saved": best_row["Bed-Days Saved"],
            "estimated_savings": best_row["Estimated Savings"],
        }

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🧠 LOS Attribution Engine")
st.caption(f"Scenario: {scenario_name}")
st.caption("A scenario tool to estimate the impact of operational interventions on length of stay and bed-days.")

if exec_mode:
    st.markdown("## Executive Summary")
    st.info(
        f"""
**Headline**  
{exec_headline}

**Impact of current scenario**  
{exec_summary}

**Recommended focus**  
{exec_action}
"""
    )

if exec_mode and auto_recommendation is not None:
    st.markdown("## Recommended Scenario")

    if auto_recommendation["estimated_savings"] > 0:
        st.success(
            f"The best preset scenario is **{auto_recommendation['scenario']}**. "
            f"It is projected to reduce mean LOS by {auto_recommendation['los_reduction']:.2f} days, "
            f"release approximately {auto_recommendation['bed_days_saved']:.0f} bed-days, "
            f"and generate estimated savings of ${auto_recommendation['estimated_savings']:,.0f}."
        )
    else:
        st.warning(
            f"None of the current preset scenarios improves performance materially. "
            f"The least harmful option is **{auto_recommendation['scenario']}**, "
            f"but it still does not generate net savings."
        )

# --------------------------------------------------
# TOP METRICS
# --------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("Episodes", len(df))
m2.metric("Mean LOS", f"{df['actual_los'].mean():.2f}")
m3.metric("Median LOS", f"{df['actual_los'].median():.2f}")
m4.metric("P90 LOS", f"{df['actual_los'].quantile(0.90):.2f}")

st.subheader("Intervention Impact")
i1, i2, i3, i4 = st.columns(4)
i1.metric("Baseline mean LOS", f"{baseline_los:.2f}")
i2.metric("LOS reduction", f"{los_reduction:.2f}")
i3.metric("Bed-days saved", f"{bed_days_saved:.0f}")
i4.metric("Estimated cost savings ($)", f"{financial_savings:,.0f}")

if exec_mode:
    st.subheader("Executive View")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Top specialty by LOS", top_specialty)
    e2.metric("Top delay driver", top_driver_label.title())
    e3.metric("Bed-days released", f"{bed_days_saved:.0f}")
    e4.metric("Estimated savings", f"${financial_savings:,.0f}")

st.markdown("### Key Insight")
if los_reduction > 0:
    st.success(
        f"Mean LOS falls by {los_reduction:.2f} days, with approximately "
        f"{int(bed_days_saved)} bed-days released and ${int(financial_savings):,} in estimated savings."
    )
elif los_reduction < 0:
    st.error(
        f"Mean LOS increases by {abs(los_reduction):.2f} days, resulting in approximately "
        f"{abs(int(bed_days_saved))} additional bed-days and ${abs(int(financial_savings)):,} in additional cost."
    )
else:
    st.info("No net change in LOS under the current scenario.")

if los_reduction < 0:
    st.warning(
        f"This scenario worsens performance. The likely reason is that the main current bottleneck is "
        f"{top_driver_label}, but the selected intervention mix is not improving that enough."
    )

if "non_clinical_delay" in baseline.columns:
    avg_non_clinical_delay = baseline["non_clinical_delay"].mean()
    st.caption(f"Average non-clinical delay: {avg_non_clinical_delay:.2f} days")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Specialty Analysis", "Scenario Comparison", "Example Patient"]
)

with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("LOS Distribution")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(df["actual_los"], bins=20)
        ax1.set_title("Actual LOS Distribution")
        ax1.set_xlabel("LOS (days)")
        ax1.set_ylabel("Patients")
        st.pyplot(fig1)
        plt.close(fig1)

    with c2:
        st.subheader("Delay Contribution")
        totals = {
            "Diagnostics": df["diagnostics"].sum(),
            "Allied": df["allied"].sum(),
            "Destination": df["destination"].sum(),
            "Weekend": df["weekend"].sum(),
            "Discharge": df["discharge"].sum(),
        }
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(totals.keys(), totals.values())
        ax2.set_title("Total Bed-Days by Delay")
        ax2.set_xlabel("Delay component")
        ax2.set_ylabel("Total bed-days")
        ax2.tick_params(axis="x", rotation=25)
        st.pyplot(fig2)
        plt.close(fig2)

with tab2:
    st.subheader("Specialty Summary")

    baseline_specialty = (
        baseline.groupby("specialty", as_index=False)
        .agg(
            episodes=("episode_id", "count"),
            baseline_mean_los=("actual_los", "mean"),
        )
    )

    intervention_specialty = (
        df.groupby("specialty", as_index=False)
        .agg(
            intervention_mean_los=("actual_los", "mean"),
            expected_mean_los=("expected_los", "mean"),
            diagnostics_days=("diagnostics", "mean"),
            allied_days=("allied", "mean"),
            destination_days=("destination", "mean"),
            weekend_days=("weekend", "mean"),
        )
    )

    specialty_summary = baseline_specialty.merge(
        intervention_specialty, on="specialty", how="outer"
    )
    specialty_summary["los_reduction"] = (
        specialty_summary["baseline_mean_los"] - specialty_summary["intervention_mean_los"]
    )

    st.dataframe(
        specialty_summary.round(2).sort_values("intervention_mean_los", ascending=False),
        width="stretch"
    )

    st.subheader("Baseline vs Intervention by Specialty")

    chart_df = specialty_summary.sort_values("intervention_mean_los", ascending=False)
    x = np.arange(len(chart_df))
    bar_width = 0.35

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(x - bar_width / 2, chart_df["baseline_mean_los"], bar_width, label="Baseline")
    ax3.bar(x + bar_width / 2, chart_df["intervention_mean_los"], bar_width, label="Intervention")
    ax3.set_xticks(x)
    ax3.set_xticklabels(chart_df["specialty"])
    ax3.set_ylabel("Mean LOS")
    ax3.set_title("Baseline vs Intervention Mean LOS")
    ax3.legend()
    st.pyplot(fig3)
    plt.close(fig3)

    st.subheader("Delay Attribution by Specialty")

    delay_summary = (
        df.groupby("specialty")
        .agg(
            mean_los=("actual_los", "mean"),
            diagnostics=("diagnostics", "mean"),
            allied=("allied", "mean"),
            destination=("destination", "mean"),
            weekend=("weekend", "mean"),
        )
        .reset_index()
    )

    def get_biggest_driver(row):
        drivers = {
            "Diagnostics": row["diagnostics"],
            "Allied": row["allied"],
            "Destination": row["destination"],
            "Weekend": row["weekend"],
        }
        return max(drivers, key=drivers.get)

    delay_summary["Biggest Driver"] = delay_summary.apply(get_biggest_driver, axis=1)

    st.dataframe(delay_summary.round(2), width="stretch")

    st.subheader("Delay Breakdown by Specialty")

    x2 = np.arange(len(delay_summary))

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.bar(x2, delay_summary["diagnostics"], label="Diagnostics")
    ax4.bar(x2, delay_summary["allied"], bottom=delay_summary["diagnostics"], label="Allied")
    ax4.bar(
        x2,
        delay_summary["destination"],
        bottom=delay_summary["diagnostics"] + delay_summary["allied"],
        label="Destination"
    )
    ax4.bar(
        x2,
        delay_summary["weekend"],
        bottom=delay_summary["diagnostics"] + delay_summary["allied"] + delay_summary["destination"],
        label="Weekend"
    )

    ax4.set_xticks(x2)
    ax4.set_xticklabels(delay_summary["specialty"])
    ax4.set_ylabel("Days")
    ax4.set_title("Delay Contribution by Specialty")
    ax4.legend()
    st.pyplot(fig4)
    plt.close(fig4)

with tab3:
    st.subheader("Scenario Comparison")

    comparison = pd.DataFrame({
        "Scenario": ["Baseline", scenario_name],
        "Mean LOS": [baseline_los, new_los],
        "Median LOS": [baseline["actual_los"].median(), df["actual_los"].median()],
        "P90 LOS": [baseline["actual_los"].quantile(0.90), df["actual_los"].quantile(0.90)],
        "Total Bed-Days": [baseline["actual_los"].sum(), df["actual_los"].sum()],
        "Estimated Cost": [
            baseline["actual_los"].sum() * cost_per_bed_day,
            df["actual_los"].sum() * cost_per_bed_day
        ],
    })

    st.dataframe(comparison.round(2), width="stretch")

    if auto_recommendation is not None:
        st.markdown("### Best Preset Option")
        st.write(
            f"Best current preset scenario: **{auto_recommendation['scenario']}**"
        )

    if scenario_mode and scenario_table is not None:
        st.subheader("Preset Scenario Ranking")
        st.dataframe(scenario_table.round(2), width="stretch", hide_index=True)
        st.caption("This compares preset packages vs baseline, not the current live sliders.")
        best_scenario = scenario_table.sort_values("Estimated Savings", ascending=False).iloc[0]
        st.success(
            f"Highest estimated savings: {best_scenario['Scenario']} "
            f"(${best_scenario['Estimated Savings']:,.0f})."
        )

    st.info(
        "Use the sliders on the left to estimate how general and specialty-specific interventions "
        "could change LOS, bed-day consumption, and financial impact."
    )

with tab4:
    st.subheader("Example Patient Attribution")

    row = df.sample(1, random_state=42).iloc[0]

    example_output = {
        "Episode ID": row["episode_id"] if "episode_id" in row.index else "N/A",
        "Specialty": row["specialty"],
        "Expected LOS": round(row["expected_los"], 2),
        "Diagnostic delay": round(row["diagnostics"], 2),
        "Allied health delay": round(row["allied"], 2),
        "Destination delay": round(row["destination"], 2),
        "Weekend-related delay": round(row["weekend"], 2),
        "Discharge process delay": round(row["discharge"], 2),
        "Actual LOS": round(row["actual_los"], 2),
    }

    if "patient_id" in row.index:
        example_output["Patient ID"] = row["patient_id"]

    if "clinical_los" in row.index:
        example_output["Clinical LOS"] = round(row["clinical_los"], 2)

    if "non_clinical_delay" in row.index:
        example_output["Non-clinical delay"] = round(row["non_clinical_delay"], 2)

    st.write(example_output)