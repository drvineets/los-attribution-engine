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
# SPECIALTY CONFIG
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

# --------------------------------------------------
# DATA QUALITY ASSESSMENT
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
            score += 0
        elif avg_optional >= 75:
            score -= 5
        elif avg_optional >= 50:
            score -= 10
        else:
            score -= 15

    results["score"] = int(max(0, min(100, score)))
    return results

# --------------------------------------------------
# FAKE DATA GENERATOR
# --------------------------------------------------
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
        else:  # GenMed1
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

# --------------------------------------------------
# SCENARIO PRESETS
# --------------------------------------------------
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
# INTERVENTION LOGIC
# --------------------------------------------------
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

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.title("LOS Simulator")

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

    # Show quality summary first
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
    st.dataframe(preview_df.head(), use_container_width=True)

    # Show optional completeness
    optional_df = pd.DataFrame(
        {
            "Column": list(quality["optional_completeness"].keys()),
            "Completeness (%)": list(quality["optional_completeness"].values()),
        }
    )

    if not optional_df.empty:
        st.markdown("### Optional Field Completeness")
        st.dataframe(optional_df, use_container_width=True)

    with st.expander("Show uploaded dataset structure"):
        st.write("Columns detected:")
        st.write(list(preview_df.columns))
        st.write("Dataset shape:")
        st.write(preview_df.shape)

    # -----------------------------
    # STRICT MODE: STOP ON CRITICAL ISSUES
    # -----------------------------
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

    # -----------------------------
    # SAFE CLEAN LOAD
    # -----------------------------
    baseline = preview_df.copy()

    baseline["admit_datetime"] = pd.to_datetime(baseline["admit_datetime"], errors="coerce")
    baseline["discharge_datetime"] = pd.to_datetime(baseline["discharge_datetime"], errors="coerce")

    baseline["actual_los"] = (
        (baseline["discharge_datetime"] - baseline["admit_datetime"]).dt.total_seconds() / (3600 * 24)
    )

    baseline["expected_los"] = baseline.groupby("specialty")["actual_los"].transform("mean")

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

    baseline["diagnostics"] = baseline["diagnostics_delay_days"]
    baseline["allied"] = baseline["allied_delay_days"]
    baseline["destination"] = baseline["destination_delay_days"]
    baseline["discharge"] = baseline["discharge_process_delay_days"]
    baseline["weekend"] = baseline["admit_datetime"].dt.dayofweek.isin([5, 6]).astype(float) * 0.3

    baseline = baseline.dropna(subset=["episode_id", "specialty", "admit_datetime", "discharge_datetime"])
    baseline = baseline[baseline["actual_los"] > 0].copy()

    if baseline.empty:
        st.error("No valid rows remain after validation and cleaning.")
        st.stop()

else:
    baseline = generate_data(n_patients, seed=42)

# --------------------------------------------------
# APPLY INTERVENTIONS
# --------------------------------------------------
for col in ["diagnostics", "allied", "destination", "discharge", "weekend"]:
    if col not in baseline.columns:
        baseline[col] = 0.0
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
# METRICS
# --------------------------------------------------
baseline_los = baseline["actual_los"].mean()
new_los = df["actual_los"].mean()
los_reduction = baseline_los - new_los
bed_days_saved = los_reduction * len(df)
financial_savings = bed_days_saved * cost_per_bed_day

st.title("🧠 LOS Attribution Engine")
st.caption(f"Scenario: {scenario_name}")
st.caption("A simple scenario tool to estimate the impact of operational interventions on length of stay and bed-days.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Episodes", len(df))
m2.metric("Mean LOS", f"{df['actual_los'].mean():.2f}")
m3.metric("Median LOS", f"{df['actual_los'].median():.2f}")
m4.metric("P90 LOS", f"{df['actual_los'].quantile(0.90):.2f}")

st.subheader("Intervention Impact")
i1, i2, i3, i4 = st.columns(4)
i1.metric("Baseline mean LOS", f"{baseline_los:.2f}")
i2.metric("LOS reduction", f"{los_reduction:.2f}")
i3.metric("Bed-days saved", f"{int(bed_days_saved)}")
i4.metric("Estimated cost savings ($)", f"{int(financial_savings):,}")

st.markdown("### Key Insight")
if los_reduction > 0:
    st.success(
        f"This scenario reduces mean LOS by {los_reduction:.2f} days across {len(df)} episodes, "
        f"saving approximately {int(bed_days_saved)} bed-days and "
        f"${int(financial_savings):,} in estimated cost."
    )
else:
    st.info("No improvement applied yet. Adjust the intervention sliders to simulate impact.")

# --------------------------------------------------
# DATA QUALITY DISPLAY
# --------------------------------------------------
if uploaded_file is not None and preview_df is not None and quality is not None:
    st.subheader("Data Quality Assessment")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Data quality score", f"{quality['score']}/100")
    q2.metric("Missing required columns", len(quality["missing_required"]))
    q3.metric("Invalid timestamps", quality["invalid_admit_datetime"] + quality["invalid_discharge_datetime"])
    q4.metric("Invalid LOS rows", quality["negative_or_zero_los"])

    if quality["score"] >= 85:
        st.success("High-quality dataset: suitable for real modelling.")
    elif quality["score"] >= 65:
        st.warning("Moderate-quality dataset: usable, but some fields need improvement.")
    else:
        st.error("Low-quality dataset: fix key data issues before using for decision-making.")

    st.subheader("Uploaded Data Preview")
    st.dataframe(preview_df.head(), use_container_width=True)

    if quality["missing_required"]:
        st.error(f"Missing required columns: {quality['missing_required']}")
        st.stop()

    warnings = []
    if quality["missing_episode_id"] > 0:
        warnings.append(f"{quality['missing_episode_id']} rows are missing episode_id.")
    if quality["missing_specialty"] > 0:
        warnings.append(f"{quality['missing_specialty']} rows are missing specialty.")
    if quality["invalid_admit_datetime"] > 0:
        warnings.append(f"{quality['invalid_admit_datetime']} rows have invalid admit_datetime.")
    if quality["invalid_discharge_datetime"] > 0:
        warnings.append(f"{quality['invalid_discharge_datetime']} rows have invalid discharge_datetime.")
    if quality["negative_or_zero_los"] > 0:
        warnings.append(f"{quality['negative_or_zero_los']} rows have zero or negative LOS.")

    if warnings:
        st.markdown("### Validation Warnings")
        for w in warnings:
            st.warning(w)

    optional_df = pd.DataFrame(
        {
            "Column": list(quality["optional_completeness"].keys()),
            "Completeness (%)": list(quality["optional_completeness"].values()),
        }
    )

    if not optional_df.empty:
        st.markdown("### Optional Field Completeness")
        st.dataframe(optional_df, use_container_width=True)

    with st.expander("Show uploaded dataset structure"):
        st.write("Columns detected:")
        st.write(list(preview_df.columns))
        st.write("Dataset shape:")
        st.write(preview_df.shape)

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
        use_container_width=True,
        column_config={
            "specialty": "Specialty",
            "episodes": "Episodes",
            "baseline_mean_los": st.column_config.NumberColumn("Baseline mean LOS", format="%.2f"),
            "intervention_mean_los": st.column_config.NumberColumn("Intervention mean LOS", format="%.2f"),
            "expected_mean_los": st.column_config.NumberColumn("Expected mean LOS", format="%.2f"),
            "diagnostics_days": st.column_config.NumberColumn("Diagnostics delay", format="%.2f"),
            "allied_days": st.column_config.NumberColumn("Allied health delay", format="%.2f"),
            "destination_days": st.column_config.NumberColumn("Destination delay", format="%.2f"),
            "weekend_days": st.column_config.NumberColumn("Weekend delay", format="%.2f"),
            "los_reduction": st.column_config.NumberColumn("LOS reduction", format="%.2f"),
        }
    )

    st.subheader("Baseline vs Intervention by Specialty")
    chart_df = specialty_summary.sort_values("intervention_mean_los", ascending=False)
    x = np.arange(len(chart_df))
    width = 0.35

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(x - width / 2, chart_df["baseline_mean_los"], width, label="Baseline")
    ax3.bar(x + width / 2, chart_df["intervention_mean_los"], width, label="Intervention")
    ax3.set_xticks(x)
    ax3.set_xticklabels(chart_df["specialty"])
    ax3.set_ylabel("Mean LOS")
    ax3.set_title("Baseline vs Intervention Mean LOS")
    ax3.legend()
    st.pyplot(fig3)

with tab3:
    st.subheader("Scenario Comparison")

    comparison = pd.DataFrame({
        "Scenario": ["Baseline", "Intervention"],
        "Mean LOS": [baseline_los, new_los],
        "Median LOS": [baseline["actual_los"].median(), df["actual_los"].median()],
        "P90 LOS": [baseline["actual_los"].quantile(0.90), df["actual_los"].quantile(0.90)],
        "Total Bed-Days": [baseline["actual_los"].sum(), df["actual_los"].sum()],
        "Estimated Cost": [baseline["actual_los"].sum() * cost_per_bed_day, df["actual_los"].sum() * cost_per_bed_day],
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        column_config={
            "Mean LOS": st.column_config.NumberColumn(format="%.2f"),
            "Median LOS": st.column_config.NumberColumn(format="%.2f"),
            "P90 LOS": st.column_config.NumberColumn(format="%.2f"),
            "Total Bed-Days": st.column_config.NumberColumn(format="%.1f"),
            "Estimated Cost": st.column_config.NumberColumn(format="$%.0f"),
        }
    )

    st.info(
        "Use the sliders on the left to estimate how general and specialty-specific interventions "
        "could change LOS, bed-day consumption, and financial impact."
    )

with tab4:
    st.subheader("Example Patient Attribution")
    row = df.sample(1, random_state=42).iloc[0]

    st.write({
        "Episode ID": row["episode_id"] if "episode_id" in row else "N/A",
        "Patient ID": row["patient_id"] if "patient_id" in row else "N/A",
        "Specialty": row["specialty"],
        "Expected LOS": round(row["expected_los"], 2),
        "Diagnostic delay": round(row["diagnostics"], 2),
        "Allied health delay": round(row["allied"], 2),
        "Destination delay": round(row["destination"], 2),
        "Weekend-related delay": round(row["weekend"], 2),
        "Discharge process delay": round(row["discharge"], 2),
        "Actual LOS": round(row["actual_los"], 2),
    })