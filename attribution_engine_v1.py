import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_PATIENTS = 300

np.random.seed(SEED)


# ============================================================
# DOMAIN CONFIG
# ============================================================

SPECIALTIES = {
    "General Medicine": {"base_los_mean": 4.5, "base_los_sd": 1.2},
    "Cardiology": {"base_los_mean": 3.8, "base_los_sd": 1.0},
    "Respiratory": {"base_los_mean": 5.0, "base_los_sd": 1.3},
    "Neurology": {"base_los_mean": 5.8, "base_los_sd": 1.4},
    "Surgery": {"base_los_mean": 3.2, "base_los_sd": 0.9},
    "Oncology": {"base_los_mean": 5.5, "base_los_sd": 1.5},
}

DESTINATIONS = [
    "Home",
    "Home with supports",
    "Rehab",
    "Aged care",
    "Transfer",
]

DIAGNOSTIC_COMPLEXITY = ["Low", "Medium", "High"]
ALLIED_NEED = ["None", "Low", "Moderate", "High"]
DESTINATION_COMPLEXITY = ["Simple", "Moderate", "Complex"]

AGE_GROUPS = ["18-44", "45-64", "65-79", "80+"]

DAY_NAMES = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bounded_normal(mean: float, sd: float, lower: float, upper: float) -> float:
    value = np.random.normal(mean, sd)
    return float(np.clip(value, lower, upper))


def weighted_choice(options: List[str], probs: List[float]) -> str:
    return str(np.random.choice(options, p=probs))


def random_date(start: str = "2025-01-01", end: str = "2025-12-31") -> pd.Timestamp:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    delta_days = (end_ts - start_ts).days
    return start_ts + pd.Timedelta(days=int(np.random.randint(0, delta_days + 1)))


def age_group_to_age(age_group: str) -> int:
    if age_group == "18-44":
        return int(np.random.randint(18, 45))
    if age_group == "45-64":
        return int(np.random.randint(45, 65))
    if age_group == "65-79":
        return int(np.random.randint(65, 80))
    return int(np.random.randint(80, 96))


# ============================================================
# FAKE DATASET GENERATOR
# ============================================================

def generate_fake_dataset(n_patients: int = N_PATIENTS) -> pd.DataFrame:
    rows: List[Dict] = []

    for i in range(1, n_patients + 1):
        patient_id = f"P{i:04d}"
        episode_id = f"E{i:04d}"

        specialty = weighted_choice(
            list(SPECIALTIES.keys()),
            [0.28, 0.14, 0.18, 0.10, 0.18, 0.12]
        )

        age_group = weighted_choice(AGE_GROUPS, [0.14, 0.22, 0.32, 0.32])
        age = age_group_to_age(age_group)

        admit_date = random_date()
        admit_day_num = admit_date.dayofweek
        admit_day_name = DAY_NAMES[admit_day_num]
        weekend_admit = admit_day_num >= 5

        frailty_score = bounded_normal(
            mean=4.0 if age >= 80 else 2.7 if age >= 65 else 1.8,
            sd=1.4,
            lower=0,
            upper=8,
        )

        comorbidity_index = bounded_normal(
            mean=4.5 if age >= 80 else 3.2 if age >= 65 else 2.0,
            sd=1.8,
            lower=0,
            upper=10,
        )

        severity_score = bounded_normal(
            mean=5.5 if specialty in {"Neurology", "Respiratory", "Oncology"} else 4.6,
            sd=1.5,
            lower=1,
            upper=10,
        )

        diagnostic_complexity = weighted_choice(
            DIAGNOSTIC_COMPLEXITY,
            [0.35, 0.45, 0.20] if specialty != "Neurology" else [0.20, 0.45, 0.35]
        )

        allied_need = weighted_choice(
            ALLIED_NEED,
            [0.18, 0.27, 0.34, 0.21] if age >= 65 else [0.30, 0.35, 0.25, 0.10]
        )

        destination = weighted_choice(
            DESTINATIONS,
            [0.42, 0.20, 0.18, 0.12, 0.08] if age >= 65 else [0.68, 0.20, 0.07, 0.01, 0.04]
        )

        destination_complexity = (
            "Complex" if destination in {"Rehab", "Aged care", "Transfer"} and np.random.rand() < 0.6
            else "Moderate" if destination in {"Rehab", "Home with supports", "Transfer"}
            else "Simple"
        )

        specialty_base = SPECIALTIES[specialty]
        expected_los = bounded_normal(
            specialty_base["base_los_mean"],
            specialty_base["base_los_sd"],
            1.5,
            10.0,
        )

        expected_los += 0.08 * frailty_score
        expected_los += 0.06 * comorbidity_index
        expected_los += 0.05 * max(0, severity_score - 4)

        if age >= 80:
            expected_los += 0.5
        elif age >= 65:
            expected_los += 0.25

        expected_los = round(expected_los, 2)

        if diagnostic_complexity == "Low":
            diagnostics_delay = bounded_normal(0.2, 0.2, 0, 0.8)
        elif diagnostic_complexity == "Medium":
            diagnostics_delay = bounded_normal(0.7, 0.4, 0, 2.0)
        else:
            diagnostics_delay = bounded_normal(1.5, 0.8, 0, 4.0)

        weekend_delay = 0.0
        if weekend_admit:
            weekend_delay += bounded_normal(0.4, 0.3, 0, 1.2)
        if admit_day_num == 4 and np.random.rand() < 0.4:
            weekend_delay += bounded_normal(0.3, 0.2, 0, 0.9)

        if allied_need == "None":
            allied_delay = bounded_normal(0.0, 0.05, 0, 0.2)
        elif allied_need == "Low":
            allied_delay = bounded_normal(0.2, 0.2, 0, 0.9)
        elif allied_need == "Moderate":
            allied_delay = bounded_normal(0.7, 0.4, 0, 2.0)
        else:
            allied_delay = bounded_normal(1.4, 0.7, 0, 3.5)

        if destination_complexity == "Simple":
            destination_delay = bounded_normal(0.1, 0.15, 0, 0.7)
        elif destination_complexity == "Moderate":
            destination_delay = bounded_normal(0.8, 0.5, 0, 2.5)
        else:
            destination_delay = bounded_normal(2.0, 1.0, 0, 5.0)

        discharge_planning_delay = bounded_normal(
            0.25 if age < 65 else 0.5,
            0.25,
            0,
            1.5,
        )

        unexplained_delay = bounded_normal(0.25, 0.35, 0, 1.5)

        total_delay = (
            diagnostics_delay
            + weekend_delay
            + allied_delay
            + destination_delay
            + discharge_planning_delay
            + unexplained_delay
        )

        actual_los = round(expected_los + total_delay, 2)

        weekend_days_in_stay = min(
            2,
            int(np.random.binomial(2, 0.45 if actual_los >= 4 else 0.22))
        )

        discharge_before_noon = bool(
            np.random.rand()
            < max(
                0.12,
                0.45
                - 0.07 * destination_delay
                - 0.05 * allied_delay
                - 0.04 * diagnostics_delay
            )
        )

        avoidable_bed_days = max(
            0.0,
            round(
                diagnostics_delay
                + allied_delay
                + destination_delay
                + weekend_delay
                + 0.5 * discharge_planning_delay,
                2,
            ),
        )

        dominant_delay = max(
            {
                "Diagnostics": diagnostics_delay,
                "Weekend": weekend_delay,
                "Allied": allied_delay,
                "Destination": destination_delay,
                "DischargePlanning": discharge_planning_delay,
                "Unexplained": unexplained_delay,
            },
            key=lambda k: {
                "Diagnostics": diagnostics_delay,
                "Weekend": weekend_delay,
                "Allied": allied_delay,
                "Destination": destination_delay,
                "DischargePlanning": discharge_planning_delay,
                "Unexplained": unexplained_delay,
            }[k],
        )

        if destination_delay >= 1.5:
            delay_cluster = "Destination-blocked"
        elif diagnostics_delay >= 1.2:
            delay_cluster = "Diagnostics-driven"
        elif allied_delay >= 1.0:
            delay_cluster = "Allied-health constrained"
        elif weekend_delay >= 0.7:
            delay_cluster = "Weekend-affected"
        else:
            delay_cluster = "Low-delay / routine"

        discharge_date = admit_date + pd.Timedelta(days=math.ceil(actual_los))

        rows.append(
            {
                "episode_id": episode_id,
                "patient_id": patient_id,
                "age": age,
                "age_group": age_group,
                "specialty": specialty,
                "admission_date": admit_date,
                "admit_day_num": admit_day_num,
                "admit_day_name": admit_day_name,
                "weekend_admit": weekend_admit,
                "frailty_score": round(frailty_score, 1),
                "comorbidity_index": round(comorbidity_index, 1),
                "severity_score": round(severity_score, 1),
                "diagnostic_complexity": diagnostic_complexity,
                "allied_need": allied_need,
                "destination": destination,
                "destination_complexity": destination_complexity,
                "expected_los": expected_los,
                "diagnostics_delay": round(diagnostics_delay, 2),
                "weekend_delay": round(weekend_delay, 2),
                "allied_delay": round(allied_delay, 2),
                "destination_delay": round(destination_delay, 2),
                "discharge_planning_delay": round(discharge_planning_delay, 2),
                "unexplained_delay": round(unexplained_delay, 2),
                "actual_los": actual_los,
                "avoidable_bed_days": avoidable_bed_days,
                "weekend_days_in_stay": weekend_days_in_stay,
                "discharge_before_noon": discharge_before_noon,
                "discharge_date": discharge_date,
                "dominant_delay": dominant_delay,
                "delay_cluster": delay_cluster,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ATTRIBUTION ENGINE
# ============================================================

ATTRIBUTION_FIELDS = [
    "diagnostics_delay",
    "weekend_delay",
    "allied_delay",
    "destination_delay",
    "discharge_planning_delay",
    "unexplained_delay",
]


def build_patient_attribution(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["reconstructed_los"] = (
        df["expected_los"]
        + df["diagnostics_delay"]
        + df["weekend_delay"]
        + df["allied_delay"]
        + df["destination_delay"]
        + df["discharge_planning_delay"]
        + df["unexplained_delay"]
    ).round(2)

    df["reconstruction_gap"] = (df["actual_los"] - df["reconstructed_los"]).round(4)

    df["largest_delay_component"] = df[ATTRIBUTION_FIELDS].idxmax(axis=1)

    label_map = {
        "diagnostics_delay": "Diagnostics",
        "weekend_delay": "Weekend",
        "allied_delay": "Allied Health",
        "destination_delay": "Destination",
        "discharge_planning_delay": "Discharge Planning",
        "unexplained_delay": "Unexplained",
    }
    df["largest_delay_component"] = df["largest_delay_component"].map(label_map)

    return df


def patient_narrative(row: pd.Series) -> str:
    components = {
        "Diagnostics": row["diagnostics_delay"],
        "Weekend": row["weekend_delay"],
        "Allied health": row["allied_delay"],
        "Destination": row["destination_delay"],
        "Discharge planning": row["discharge_planning_delay"],
        "Unexplained": row["unexplained_delay"],
    }

    sorted_components = sorted(components.items(), key=lambda x: x[1], reverse=True)
    top_3 = [f"{name} +{value:.1f}d" for name, value in sorted_components if value > 0][:3]

    return (
        f"Episode {row['episode_id']} | {row['specialty']} | LOS {row['actual_los']:.1f}d\n"
        f"Expected {row['expected_los']:.1f}d | Avoidable {row['avoidable_bed_days']:.1f}d\n"
        f"Main contributors: {', '.join(top_3) if top_3 else 'None'}\n"
        f"Dominant delay pattern: {row['delay_cluster']}"
    )


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "Metric": [
                "Episodes",
                "Mean LOS",
                "Median LOS",
                "P90 LOS",
                "Mean Expected LOS",
                "Mean Avoidable Bed Days",
                "% Weekend Admit",
                "% Discharge Before Noon",
            ],
            "Value": [
                len(df),
                round(df["actual_los"].mean(), 2),
                round(df["actual_los"].median(), 2),
                round(df["actual_los"].quantile(0.90), 2),
                round(df["expected_los"].mean(), 2),
                round(df["avoidable_bed_days"].mean(), 2),
                round(100 * df["weekend_admit"].mean(), 1),
                round(100 * df["discharge_before_noon"].mean(), 1),
            ],
        }
    )
    return summary


def specialty_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("specialty")
        .agg(
            episodes=("episode_id", "count"),
            mean_los=("actual_los", "mean"),
            median_los=("actual_los", "median"),
            mean_expected=("expected_los", "mean"),
            mean_avoidable=("avoidable_bed_days", "mean"),
            pct_discharge_before_noon=("discharge_before_noon", "mean"),
        )
        .reset_index()
    )

    out["mean_los"] = out["mean_los"].round(2)
    out["median_los"] = out["median_los"].round(2)
    out["mean_expected"] = out["mean_expected"].round(2)
    out["mean_avoidable"] = out["mean_avoidable"].round(2)
    out["pct_discharge_before_noon"] = (100 * out["pct_discharge_before_noon"]).round(1)

    return out.sort_values("mean_los", ascending=False)


def delay_component_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ATTRIBUTION_FIELDS:
        rows.append(
            {
                "component": col,
                "mean_days": round(df[col].mean(), 2),
                "median_days": round(df[col].median(), 2),
                "p90_days": round(df[col].quantile(0.90), 2),
                "total_bed_days": round(df[col].sum(), 1),
            }
        )

    out = pd.DataFrame(rows).sort_values("total_bed_days", ascending=False)
    return out


def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby("delay_cluster")
        .agg(
            episodes=("episode_id", "count"),
            mean_los=("actual_los", "mean"),
            mean_avoidable=("avoidable_bed_days", "mean"),
        )
        .reset_index()
    )
    out["pct_of_cohort"] = (100 * out["episodes"] / len(df)).round(1)
    out["mean_los"] = out["mean_los"].round(2)
    out["mean_avoidable"] = out["mean_avoidable"].round(2)
    return out.sort_values("episodes", ascending=False)


def top_delay_cases(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = [
        "episode_id",
        "specialty",
        "age",
        "actual_los",
        "expected_los",
        "avoidable_bed_days",
        "diagnostics_delay",
        "weekend_delay",
        "allied_delay",
        "destination_delay",
        "discharge_planning_delay",
        "dominant_delay",
        "delay_cluster",
    ]
    return df.sort_values("avoidable_bed_days", ascending=False)[cols].head(n)


def plot_los_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.hist(df["actual_los"], bins=25)
    plt.title("Actual LOS Distribution")
    plt.xlabel("Length of stay (days)")
    plt.ylabel("Episodes")
    plt.tight_layout()
    plt.show()


def plot_delay_components(delay_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(delay_df["component"], delay_df["total_bed_days"])
    plt.title("Total Bed-Days Attributed to Delay Components")
    plt.xlabel("Delay component")
    plt.ylabel("Total bed-days")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_expected_vs_actual(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["expected_los"], df["actual_los"], alpha=0.7)
    max_val = max(df["expected_los"].max(), df["actual_los"].max()) + 1
    plt.plot([0, max_val], [0, max_val], linestyle="--")
    plt.title("Expected vs Actual LOS")
    plt.xlabel("Expected LOS")
    plt.ylabel("Actual LOS")
    plt.tight_layout()
    plt.show()


def plot_cluster_counts(cluster_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    plt.bar(cluster_df["delay_cluster"], cluster_df["episodes"])
    plt.title("Delay Pattern Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Episodes")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    print("\n=== GENERATING FAKE DATASET ===")
    df = generate_fake_dataset(300)

    print("\n=== BUILDING ATTRIBUTION ENGINE ===")
    df = build_patient_attribution(df)

    print("\n=== COHORT SUMMARY ===")
    print(cohort_summary(df).to_string(index=False))

    print("\n=== SPECIALTY SUMMARY ===")
    print(specialty_summary(df).to_string(index=False))

    print("\n=== DELAY COMPONENT SUMMARY ===")
    delay_df = delay_component_summary(df)
    print(delay_df.to_string(index=False))

    print("\n=== DELAY CLUSTER SUMMARY ===")
    cluster_df = cluster_summary(df)
    print(cluster_df.to_string(index=False))

    print("\n=== TOP 10 EPISODES BY AVOIDABLE BED-DAYS ===")
    print(top_delay_cases(df, 10).to_string(index=False))

    print("\n=== EXAMPLE PATIENT-LEVEL NARRATIVES ===")
    sample_ids = df.sample(5, random_state=SEED).index
    for idx in sample_ids:
        print("\n" + patient_narrative(df.loc[idx]))

    print("\n=== EXPORTING CSV ===")
    df.to_csv("attribution_engine_fake_dataset_v1.csv", index=False)
    print("Saved: attribution_engine_fake_dataset_v1.csv")

    print("\n=== PLOTTING ===")
    plot_los_distribution(df)
    plot_delay_components(delay_df)
    plot_expected_vs_actual(df)
    plot_cluster_counts(cluster_df)

    print("\nDone.")


if __name__ == "__main__":
    main()