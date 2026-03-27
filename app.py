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
# DATA GENERATOR
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

    # global reductions
    out["diagnostics"] *= (1 - global_diag_reduction / 100)
    out["allied"] *= (1 - global_allied_reduction / 100)
    out["destination"] *= (1 - global_dest_reduction / 100)
    out["weekend"] *= (1 - global_weekend_reduction / 100)

    # neurology + respiratory especially benefit from faster diagnostics
    neuro_resp_mask = out["specialty"].isin(["Neurology", "Respiratory"])
    out.loc[neuro_resp_mask, "diagnostics"] *= (1 - neuro_resp_diag_bonus / 100)

    # geriatrics especially benefits from destination and allied support
    geri_mask = out["specialty"] == "Geriatrics"
    out.loc[geri_mask, "destination"] *= (1 - geri_dest_bonus / 100)
    out.loc[geri_mask, "allied"] *= (1 - geri_allied_bonus / 100)

    # weekend discharge process mostly reduces weekend delay
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

preset = SCENARIOS[scenario]

st.sidebar.header("Population")
n_patients = st.sidebar.slider("Number of simulated patients", 50, 1000, 300)

selected_specialties = st.sidebar.multiselect(
    "Specialties included",
    SPECIALTIES,
    default=SPECIALTIES,
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
# DATA
# --------------------------------------------------
baseline = generate_data(n_patients, seed=42)

if selected_specialties:
    baseline = baseline[baseline["specialty"].isin(selected_specialties)].copy()

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

if len(df) == 0:
    st.warning("No specialties selected.")
    st.stop()

# --------------------------------------------------
# METRICS
# --------------------------------------------------
baseline_los = baseline["actual_los"].mean()
new_los = df["actual_los"].mean()
los_reduction = baseline_los - new_los
bed_days_saved = los_reduction * len(df)

st.title("🧠 LOS Attribution Engine")
st.caption("A simple scenario tool to estimate the impact of operational interventions on length of stay and bed-days.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Episodes", len(df))
m2.metric("Mean LOS", f"{df['actual_los'].mean():.2f}")
m3.metric("Median LOS", f"{df['actual_los'].median():.2f}")
m4.metric("P90 LOS", f"{df['actual_los'].quantile(0.90):.2f}")

st.subheader("Intervention Impact")
i1, i2, i3 = st.columns(3)
i1.metric("Baseline mean LOS", f"{baseline_los:.2f}")
i2.metric("LOS reduction", f"{los_reduction:.2f}")
i3.metric("Bed-days saved", f"{int(bed_days_saved)}")

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
            episodes=("patient_id", "count"),
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
    })

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
        column_config={
            "Mean LOS": st.column_config.NumberColumn(format="%.2f"),
            "Median LOS": st.column_config.NumberColumn(format="%.2f"),
            "P90 LOS": st.column_config.NumberColumn(format="%.2f"),
            "Total Bed-Days": st.column_config.NumberColumn(format="%.1f"),
        }
    )

    st.info(
        "Use the sliders on the left to estimate how general and specialty-specific interventions "
        "could change LOS and bed-day consumption."
    )

with tab4:
    st.subheader("Example Patient Attribution")

    row = df.sample(1, random_state=42).iloc[0]

    st.write({
        "Patient ID": row["patient_id"],
        "Specialty": row["specialty"],
        "Expected LOS": round(row["expected_los"], 2),
        "Diagnostic delay": round(row["diagnostics"], 2),
        "Allied health delay": round(row["allied"], 2),
        "Destination delay": round(row["destination"], 2),
        "Weekend-related delay": round(row["weekend"], 2),
        "Discharge process delay": round(row["discharge"], 2),
        "Actual LOS": round(row["actual_los"], 2),
    })