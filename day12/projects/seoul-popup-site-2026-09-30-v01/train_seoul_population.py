#!/usr/bin/env python3
"""Backtest and fit total-population and 20–39 population models from prepared history."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import popup_modeling as pm

ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "data" / "processed" / "seoul_population_training.csv"
OUT = ROOT / "outputs" / "modeling" / "three_year_run"


def _feature_importance(history_path: Path, out_dir: Path, target: str, label: str) -> pd.DataFrame:
    pm.TARGET = target
    frame = pm._read_history(history_path)
    holdout_start = pm._latest_complete_week(frame, pm._local_today())
    if holdout_start is None:
        raise RuntimeError("완전한 마지막 테스트 주를 찾을 수 없습니다.")
    train = frame[frame["week_start"].lt(holdout_start)].copy()
    test = frame[frame["week_start"].eq(holdout_start)].copy()
    x_train, x_test = pm._features(train), pm._features(test)
    model = pm._build_quantile_model(x_train, train[target].astype(float), 0.5).fit(x_train, train[target].astype(float))
    importance = permutation_importance(
        model,
        x_test,
        test[target].astype(float),
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=42,
        n_jobs=1,
    )
    result = pd.DataFrame({
        "feature": x_test.columns,
        "mae_increase_when_shuffled": importance.importances_mean,
        "std": importance.importances_std,
        "target": label,
    }).sort_values("mae_increase_when_shuffled", ascending=False)
    result.to_csv(out_dir / f"feature_importance_{label}.csv", index=False, encoding="utf-8-sig")
    return result


def _effect_tables(frame: pd.DataFrame, out_dir: Path) -> None:
    data = frame.copy()
    data["month"] = data["observed_at"].dt.month
    month = data.groupby("month").agg(
        rows=("footfall_total", "size"),
        mean_population=("footfall_total", "mean"),
        mean_population_20_39=("population_20_39", "mean"),
    ).reset_index()
    grand_mean = data["footfall_total"].mean()
    month["index_vs_all_months"] = month["mean_population"] / grand_mean if grand_mean else np.nan
    month.to_csv(out_dir / "monthly_comparison.csv", index=False, encoding="utf-8-sig")

    # Match calendar labels to ordinary dates in the same area, month, weekday and hour.
    data["calendar_group"] = np.select(
        [data["temporary_holiday"].eq(1), data["holiday"].eq(1), data["holiday_eve"].eq(1)],
        ["temporary_holiday", "public_holiday", "holiday_eve"],
        default="ordinary_day",
    )
    keys = ["venue_id", "month", "weekday", "hour"]
    ordinary_reference = (
        data[data["calendar_group"].eq("ordinary_day")]
        .groupby(keys)[["footfall_total", "population_20_39"]]
        .mean()
        .rename(columns={
            "footfall_total": "ordinary_matched_total",
            "population_20_39": "ordinary_matched_20_39",
        })
        .reset_index()
    )
    data = data.merge(ordinary_reference, on=keys, how="left", validate="many_to_one")
    data["uplift_total"] = data["footfall_total"] - data["ordinary_matched_total"]
    data["uplift_20_39"] = data["population_20_39"] - data["ordinary_matched_20_39"]
    effects = data.groupby("calendar_group").agg(
        rows=("footfall_total", "size"),
        mean_population=("footfall_total", "mean"),
        mean_population_20_39=("population_20_39", "mean"),
        matched_uplift_total=("uplift_total", "mean"),
        matched_uplift_20_39=("uplift_20_39", "mean"),
    ).reset_index()
    effects.to_csv(out_dir / "calendar_comparison.csv", index=False, encoding="utf-8-sig")


def _fit_final_models(history_path: Path, out_dir: Path) -> None:
    frame_all = pd.read_csv(history_path, encoding="utf-8-sig", parse_dates=["observed_at"])
    for target, label in (("footfall_total", "total"), ("population_20_39", "age20_39")):
        pm.TARGET = target
        frame = pm._read_history(history_path)
        x = pm._features(frame)
        y = frame[target].astype(float)
        models = {
            "quantile_q10": pm._build_quantile_model(x, y, 0.1).fit(x, y),
            "quantile_q50": pm._build_quantile_model(x, y, 0.5).fit(x, y),
            "quantile_q90": pm._build_quantile_model(x, y, 0.9).fit(x, y),
            "poisson_mean": pm._build_poisson_model(x, y).fit(x, y),
        }
        artifact = {
            "target": target,
            "models": models,
            "feature_columns": list(x.columns),
            "training_start": frame[pm.TIME_COL].min().isoformat(),
            "training_end": frame[pm.TIME_COL].max().isoformat(),
            "rows": len(frame),
            "areas": sorted(frame["venue_id"].dropna().unique().tolist()),
            "label_note": "Seoul 250m living-population area sum; not popup entrants or sales",
        }
        joblib.dump(artifact, out_dir / f"seoul_population_{label}.joblib", compress=3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.history, encoding="utf-8-sig", parse_dates=["observed_at"])
    if frame.empty:
        raise RuntimeError("학습 자료가 비어 있습니다.")

    # Run the established chronological rolling validation and untouched latest complete-week holdout.
    for target, label in (("footfall_total", "total"), ("population_20_39", "age20_39")):
        source = args.history
        if target != "footfall_total":
            alternate = args.out_dir / "_age20_39_history.csv"
            copy = frame.copy()
            copy["footfall_total"] = copy[target]
            copy.to_csv(alternate, index=False, encoding="utf-8-sig")
            source = alternate
        result = subprocess.run([
            sys.executable, str(ROOT / "popup_modeling.py"), "backtest",
            "--history", str(source), "--out-dir", str(args.out_dir / f"backtest_{label}"),
        ], check=False, cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"{label} 정답값의 백테스트가 실패했습니다.")
        _feature_importance(source, args.out_dir, "footfall_total", label)
        if target != "footfall_total":
            alternate.unlink(missing_ok=True)

    _effect_tables(frame, args.out_dir)
    _fit_final_models(args.history, args.out_dir)
    backtests = {
        label: json.loads((args.out_dir / f"backtest_{label}" / "backtest_metrics.json").read_text(encoding="utf-8"))
        for label in ("total", "age20_39")
    }
    summary = {
        "period_start": frame["observed_at"].min().isoformat(),
        "period_end": frame["observed_at"].max().isoformat(),
        "rows": int(len(frame)),
        "areas": sorted(frame["venue_id"].dropna().unique().tolist()),
        "holdout_policy": "most recent complete week is excluded from its own training run",
        "weather_source": "Open-Meteo historical forecast model archive",
        "included_features": ["area proxy", "weekday", "hour", "month and seasonal cycles", "weekend", "public holiday", "holiday eve", "temporary holiday", "temperature", "precipitation", "wind"],
        "not_included_yet": ["specific venue address", "university distance", "commercial-area intensity", "nearby popup/event records", "air quality", "actual popup entrants or sales"],
        "interpretation_limit": "Area living population is a proxy label, not popup entrants; calendar effects are observational comparisons, not causal claims.",
        "models_refit_after_backtest": "After each untouched holdout evaluation, final artifacts are fit on all supplied history for future prediction.",
        "backtests": {
            label: {
                "holdout_week": result.get("holdout_week"),
                "metrics": result.get("metrics", {}),
                "rolling_validation": result.get("rolling_validation", {}).get("summary", {}),
                "rolling_best_model": result.get("rolling_validation", {}).get("lowest_rolling_mae_model"),
            }
            for label, result in backtests.items()
        },
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"최종 3년 모델·검증 결과: {args.out_dir.resolve()}")
    print(f"시간대 행 {len(frame):,}개, 권역 {frame['venue_id'].nunique()}개")
    print("최신 완전 주를 최종 시험 구간으로 두고 과거 롤링 검증을 완료한 뒤 전체 기간으로 배포용 모델을 다시 학습했습니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, RuntimeError, pd.errors.ParserError) as exc:
        print(f"학습 실패: {exc}", file=sys.stderr)
        raise SystemExit(2)
