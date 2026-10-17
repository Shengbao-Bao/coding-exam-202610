#!/usr/bin/env python3
"""Reproduce node-sync freshness analysis for job_executions.csv."""
from pathlib import Path

import pandas as pd


DATA = Path(__file__).with_name("job_executions.csv")
FAILURE_COST_HOURS = 120
WAIT_COST_HOURS = 12
TRAIN_FRACTION = 0.70


def summarize(frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Estimate a hard gate that holds jobs when node state is too old."""
    held = frame["sync_age_min"] > threshold
    failures = frame["execution_ok"] == 0
    successes = frame["execution_ok"] == 1
    n_held = int(held.sum())
    avoided = int((held & failures).sum())
    false_holds = int((held & successes).sum())
    remaining = frame.loc[~held, "execution_ok"]
    return {
        "jobs": len(frame),
        "baseline_failures": int(failures.sum()),
        "held": n_held,
        "avoided_failures": avoided,
        "successful_jobs_held": false_holds,
        "failures_remaining": int((~held & failures).sum()),
        "baseline_failure_rate": float(failures.mean()),
        "remaining_failure_rate": float((remaining == 0).mean()) if len(remaining) else float("nan"),
        "net_hours_estimate": avoided * FAILURE_COST_HOURS - n_held * WAIT_COST_HOURS,
    }


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["submitted_at", "node_synced_at"])
    df["sync_age_min"] = (
        df["submitted_at"] - df["node_synced_at"]
    ).dt.total_seconds() / 60
    print(f"Rows: {len(df)}")
    print("Outcome counts (including missing):")
    print(df["execution_ok"].value_counts(dropna=False).to_string())
    print("\nMissing outcomes are excluded from success-rate and policy evaluation.")
    print("Sync age (minutes) by observed outcome:")
    print(df.groupby("execution_ok", dropna=False)["sync_age_min"].agg(["count", "median", "mean"]).to_string())

    observed = df.dropna(subset=["execution_ok"]).sort_values("submitted_at").reset_index(drop=True)
    split = int(len(observed) * TRAIN_FRACTION)
    train, validation = observed.iloc[:split], observed.iloc[split:]
    print(f"\nChronological split: train={len(train)}, validation={len(validation)}")
    print("Threshold sweep; choose on train, report validation:")
    rows = []
    for threshold in range(0, 66, 5):
        rows.append({"threshold_min": threshold, **summarize(train, threshold)})
    sweep = pd.DataFrame(rows)
    chosen = float(sweep.sort_values(["net_hours_estimate", "threshold_min"], ascending=[False, True]).iloc[0]["threshold_min"])
    print(sweep[["threshold_min", "held", "avoided_failures", "successful_jobs_held", "net_hours_estimate"]].to_string(index=False))
    print(f"\nChosen threshold from training split: {chosen:g} minutes")
    print("Validation estimate:")
    print(pd.Series(summarize(validation, chosen)).to_string())
    print("\nFull observed-data reference (not an out-of-sample estimate):")
    print(pd.Series(summarize(observed, chosen)).to_string())


if __name__ == "__main__":
    main()
