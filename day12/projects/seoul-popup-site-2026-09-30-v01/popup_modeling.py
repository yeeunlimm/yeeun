#!/usr/bin/env python3
"""Time-aware footfall backtesting, quantile forecasts, and candidate comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "footfall_total"
TIME_COL = "observed_at"
OPTIONAL_NUMERIC = [
    "forecast_temperature_c",
    "forecast_precipitation_mm",
    "forecast_wind_speed_mps",
    "forecast_lead_hours",
    "holiday",
    "nearby_event_count",
    "nearby_university_count",
    "commercial_area_index",
    "transit_access_score",
]
CAT_FEATURES = ["venue_id", "spatial_unit"]
MIN_MODEL_ROWS = 200
MIN_MODEL_WEEKS = 8


def _local_today() -> date:
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _read_history(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"과거 유동인구 파일을 찾지 못했습니다: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    input_rows = len(frame)
    required = {"venue_id", TIME_COL, TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"필수 열이 없습니다: {', '.join(missing)}")

    frame["venue_id"] = frame["venue_id"].astype("string").str.strip()
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    valid = (
        frame["venue_id"].notna()
        & frame["venue_id"].ne("")
        & frame[TIME_COL].notna()
        & frame[TARGET].notna()
        & np.isfinite(frame[TARGET])
        & frame[TARGET].ge(0)
    )
    rows_removed = int((~valid).sum())
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError("유효한 실제 유동인구 관측값이 없습니다.")

    duplicate_key = frame.duplicated(["venue_id", TIME_COL], keep=False)
    if duplicate_key.any():
        examples = frame.loc[duplicate_key, ["venue_id", TIME_COL]].head(3)
        raise ValueError(
            "장소·시각이 중복된 행이 있습니다. 먼저 같은 집계 단위로 정리해 주세요. "
            f"예: {examples.to_dict(orient='records')}"
        )

    frame["week_start"] = (frame[TIME_COL] - pd.to_timedelta(frame[TIME_COL].dt.weekday, unit="D")).dt.normalize()
    frame["weekday"] = frame[TIME_COL].dt.weekday
    frame["hour"] = frame[TIME_COL].dt.hour
    frame["month"] = frame[TIME_COL].dt.month
    frame["month_sin"] = np.sin(2 * np.pi * (frame["month"] - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (frame["month"] - 1) / 12)
    frame["is_weekend"] = frame["weekday"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["weekday_sin"] = np.sin(2 * np.pi * frame["weekday"] / 7)
    frame["weekday_cos"] = np.cos(2 * np.pi * frame["weekday"] / 7)

    if "spatial_unit" not in frame:
        frame["spatial_unit"] = ""
    for column in OPTIONAL_NUMERIC + ["age20_share", "age30_share"]:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ("age20_share", "age30_share"):
        if column in frame and frame[column].dropna().gt(1).any():
            # 파일이 0~100 비율로 제공되는 경우 0~1로 통일합니다.
            frame[column] = frame[column] / 100
    frame = frame.sort_values(TIME_COL).reset_index(drop=True)
    frame.attrs["quality"] = {"input_rows": int(input_rows), "valid_rows": int(len(frame)), "removed_rows": rows_removed}
    return frame


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    result["venue_id"] = frame["venue_id"].astype("string").fillna("unknown")
    result["spatial_unit"] = frame["spatial_unit"].astype("string").fillna("unknown")
    for column in [
        "weekday", "hour", "month", "month_sin", "month_cos", "is_weekend", "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos", *OPTIONAL_NUMERIC,
    ]:
        result[column] = pd.to_numeric(frame[column], errors="coerce") if column in frame else np.nan
    # 같은 시점의 관측 연령 비율은 예측 때 미리 알 수 없는 값일 수 있어 입력에서 제외합니다.
    return result


def _prepare_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"venue_id", "forecast_at"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"미래 시나리오에 필수 열이 없습니다: {', '.join(missing)}")
    result = frame.copy()
    result["venue_id"] = result["venue_id"].astype("string").str.strip()
    result["forecast_at"] = pd.to_datetime(result["forecast_at"], errors="coerce")
    result = result.dropna(subset=["venue_id", "forecast_at"]).copy()
    result["observed_at"] = result["forecast_at"]
    if "spatial_unit" not in result:
        result["spatial_unit"] = ""
    result["weekday"] = result["forecast_at"].dt.weekday
    result["hour"] = result["forecast_at"].dt.hour
    result["month"] = result["forecast_at"].dt.month
    result["month_sin"] = np.sin(2 * np.pi * (result["month"] - 1) / 12)
    result["month_cos"] = np.cos(2 * np.pi * (result["month"] - 1) / 12)
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["weekday_sin"] = np.sin(2 * np.pi * result["weekday"] / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["weekday"] / 7)
    for column in OPTIONAL_NUMERIC + ["resident_age20_share", "resident_age30_share"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("resident_age20_share", "resident_age30_share"):
        if column in result and result[column].dropna().gt(1).any():
            result[column] = result[column] / 100
    return result


def _build_quantile_model(x: pd.DataFrame, y: pd.Series, quantile: float) -> Pipeline:
    categorical = [column for column in CAT_FEATURES if column in x]
    numeric = [column for column in x.columns if column not in categorical]
    prep = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    leaf = max(5, min(20, len(y) // 20))
    regressor = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=leaf,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("features", prep), ("regressor", regressor)])


def _build_poisson_model(x: pd.DataFrame, y: pd.Series) -> Pipeline:
    categorical = [column for column in CAT_FEATURES if column in x]
    numeric = [column for column in x.columns if column not in categorical]
    prep = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("fill", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )
    leaf = max(5, min(20, len(y) // 20))
    regressor = HistGradientBoostingRegressor(
        loss="poisson",
        max_iter=120,
        max_leaf_nodes=15,
        min_samples_leaf=leaf,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("features", prep), ("regressor", regressor)])


def _fit_quantiles(train: pd.DataFrame) -> tuple[dict[float, Pipeline], pd.DataFrame]:
    x = _features(train)
    y = train[TARGET].astype(float)
    models = {q: _build_quantile_model(x, y, q).fit(x, y) for q in (0.1, 0.5, 0.9)}
    return models, x


def _predict_quantiles(models: dict[float, Pipeline], frame: pd.DataFrame) -> np.ndarray:
    x = _features(frame)
    values = np.column_stack([models[q].predict(x) for q in (0.1, 0.5, 0.9)])
    # 표본이 적으면 분위수 곡선이 교차할 수 있어 순서대로 정렬합니다.
    values = np.sort(values, axis=1)
    return np.maximum(values, 0)


def _apply_interval_margin(bounds: np.ndarray, margin: float | None) -> np.ndarray:
    adjusted = np.asarray(bounds, dtype=float).copy()
    if margin is not None and np.isfinite(margin) and margin > 0:
        adjusted[:, 0] = np.maximum(0, adjusted[:, 0] - margin)
        adjusted[:, 2] = adjusted[:, 2] + margin
    return adjusted


def _fit_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    lookup_site_day_hour = train.groupby(["venue_id", "weekday", "hour"])[TARGET].median()
    lookup_site_hour = train.groupby(["venue_id", "hour"])[TARGET].median()
    lookup_hour = train.groupby("hour")[TARGET].median()
    overall = float(train[TARGET].median())
    output: list[float] = []
    for row in test[["venue_id", "weekday", "hour"]].itertuples(index=False, name=None):
        venue, weekday, hour = row
        output.append(
            float(lookup_site_day_hour.get((venue, weekday, hour),
                  lookup_site_hour.get((venue, hour), lookup_hour.get(hour, overall))))
        )
    return np.asarray(output)


def _baseline_ranges(history: pd.DataFrame, scenarios: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return historical quantile scenarios without describing them as ML intervals."""
    def quantiles(group: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        return group.groupby(keys)[TARGET].quantile([0.1, 0.5, 0.9]).unstack(-1)

    by_site_day_hour = quantiles(history, ["venue_id", "weekday", "hour"])
    by_site_hour = quantiles(history, ["venue_id", "hour"])
    by_weekday_hour = quantiles(history, ["weekday", "hour"])
    by_hour = quantiles(history, ["hour"])
    global_values = history[TARGET].quantile([0.1, 0.5, 0.9])
    predictions: list[list[float]] = []
    bases: list[str] = []
    for row in scenarios[["venue_id", "weekday", "hour"]].itertuples(index=False, name=None):
        venue, weekday, hour = row
        if (venue, weekday, hour) in by_site_day_hour.index:
            values, basis = by_site_day_hour.loc[(venue, weekday, hour)], "site_weekday_hour_history"
        elif (venue, hour) in by_site_hour.index:
            values, basis = by_site_hour.loc[(venue, hour)], "site_hour_history"
        elif (weekday, hour) in by_weekday_hour.index:
            values, basis = by_weekday_hour.loc[(weekday, hour)], "all_sites_weekday_hour_history_low_confidence"
        elif hour in by_hour.index:
            values, basis = by_hour.loc[hour], "all_sites_hour_history_low_confidence"
        else:
            values, basis = global_values, "all_history_low_confidence"
        predictions.append([float(values[0.1]), float(values[0.5]), float(values[0.9])])
        bases.append(basis)
    return np.maximum(np.asarray(predictions, dtype=float), 0), bases


def _fit_poisson(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    x_train, x_test = _features(train), _features(test)
    model = _build_poisson_model(x_train, train[TARGET].astype(float))
    model.fit(x_train, train[TARGET].astype(float))
    return np.maximum(model.predict(x_test), 0)


def _rolling_cv(frame: pd.DataFrame, holdout_start: pd.Timestamp, min_rows: int, min_weeks: int) -> dict[str, Any]:
    """Evaluate completed weeks chronologically, training only on earlier observations."""
    earlier = frame[frame["week_start"].lt(holdout_start)].copy()
    fold_starts = sorted(earlier["week_start"].unique())
    fold_results: list[dict[str, Any]] = []
    interval_miss_distances: list[float] = []
    for raw_start in fold_starts:
        fold_start = pd.Timestamp(raw_start)
        validation = earlier[earlier["week_start"].eq(fold_start)].copy()
        if validation[TIME_COL].dt.normalize().nunique() != 7:
            continue
        training = earlier[earlier["week_start"].lt(fold_start)].copy()
        training_weeks = _complete_week_count(training)
        if training.empty:
            continue
        actual = validation[TARGET].to_numpy(dtype=float)
        holiday_type = _holiday_week(validation)
        fold_type = "holiday_week" if holiday_type is True else "regular_week" if holiday_type is False else "holiday_unknown"
        comparison_eligible = len(training) >= min_rows and training_weeks >= min_weeks
        baseline = _fit_baseline(training, validation)
        fold_results.append({
            "week": fold_start.date().isoformat(), "type": fold_type, "model": "baseline",
            "comparison_eligible": comparison_eligible, **_metric(actual, baseline),
        })
        if not comparison_eligible:
            continue
        try:
            poisson = _fit_poisson(training, validation)
            fold_results.append({"week": fold_start.date().isoformat(), "type": fold_type, "model": "poisson", **_metric(actual, poisson)})
            quantile_models, _ = _fit_quantiles(training)
            interval = _predict_quantiles(quantile_models, validation)
            interval_miss_distances.extend(
                np.maximum(np.maximum(interval[:, 0] - actual, actual - interval[:, 2]), 0).tolist()
            )
            fold_results.append({
                "week": fold_start.date().isoformat(), "type": fold_type, "model": "quantile_regression",
                **_metric(actual, interval[:, 1]), **_interval_metric(actual, interval[:, 0], interval[:, 2]),
            })
        except (ValueError, TypeError) as exc:
            fold_results.append({"week": fold_start.date().isoformat(), "type": fold_type, "model": "ml_error", "message": str(exc)})

    summaries: dict[str, Any] = {}
    for model in ("baseline", "poisson", "quantile_regression"):
        rows = [row for row in fold_results if row["model"] == model]
        if model == "baseline" and any(row.get("comparison_eligible") for row in rows):
            rows = [row for row in rows if row.get("comparison_eligible")]
        if rows:
            summaries[model] = {
                "fold_count": len(rows),
                "mean_mae": float(np.mean([row["mae"] for row in rows])),
                "mean_wape": float(np.mean([row["wape"] for row in rows if row.get("wape") is not None]))
                if any(row.get("wape") is not None for row in rows) else None,
            }
            if model == "quantile_regression":
                summaries[model]["mean_coverage_80"] = float(np.mean([row["coverage_80"] for row in rows]))
                summaries[model]["mean_interval_width"] = float(np.mean([row["mean_width"] for row in rows]))

    by_type: dict[str, Any] = {}
    for fold_type in ("holiday_week", "regular_week", "holiday_unknown"):
        by_type[fold_type] = {}
        for model in ("baseline", "poisson", "quantile_regression"):
            rows = [row for row in fold_results if row["type"] == fold_type and row["model"] == model]
            if model == "baseline" and any(row.get("comparison_eligible") for row in rows):
                rows = [row for row in rows if row.get("comparison_eligible")]
            if rows:
                by_type[fold_type][model] = {
                    "fold_count": len(rows),
                    "mean_mae": float(np.mean([row["mae"] for row in rows])),
                    "mean_wape": float(np.mean([row["wape"] for row in rows if row.get("wape") is not None]))
                    if any(row.get("wape") is not None for row in rows) else None,
                    **({
                        "mean_coverage_80": float(np.mean([row["coverage_80"] for row in rows])),
                        "mean_interval_width": float(np.mean([row["mean_width"] for row in rows])),
                    } if model == "quantile_regression" else {}),
                }
    candidates = {name: summary["mean_mae"] for name, summary in summaries.items()}
    quantile_fold_count = summaries.get("quantile_regression", {}).get("fold_count", 0)
    margin = None
    if len(interval_miss_distances) >= 30 and quantile_fold_count >= 3:
        ordered_misses = np.sort(np.asarray(interval_miss_distances, dtype=float))
        rank = min(len(ordered_misses) - 1, max(0, int(np.ceil((len(ordered_misses) + 1) * 0.8)) - 1))
        margin = float(ordered_misses[rank])
    return {
        "folds": fold_results,
        "summary": summaries,
        "holiday_vs_regular": by_type,
        "lowest_rolling_mae_model": min(candidates, key=candidates.get) if candidates else None,
        "quantile_interval_margin_80": margin,
        "quantile_calibration_rows": len(interval_miss_distances),
        "note": "모델 선택 참고값이며, 가장 최근 완전 주 테스트는 선택/조정에 사용하지 않습니다.",
    }


def _metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    error = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "wape": float(np.abs(error).sum() / denominator) if denominator else None,
    }


def _interval_metric(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    return {
        "coverage_80": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width": float(np.mean(upper - lower)),
    }


def _latest_complete_week(frame: pd.DataFrame, as_of: date) -> pd.Timestamp | None:
    # 완전히 지난 월~일 주 가운데 실제 관측이 7일 모두 있는 가장 최근 주를 고릅니다.
    latest_closed_sunday = pd.Timestamp(as_of) - pd.to_timedelta((pd.Timestamp(as_of).weekday() + 1) % 7 or 7, unit="D")
    weeks = sorted(frame["week_start"].unique(), reverse=True)
    for week in weeks:
        week_start = pd.Timestamp(week)
        week_end = week_start + pd.Timedelta(days=6)
        if week_end > latest_closed_sunday:
            continue
        week_rows = frame[frame["week_start"].eq(week_start)]
        if week_rows[TARGET].notna().all() and week_rows[TIME_COL].dt.normalize().nunique() == 7:
            return week_start
    return None


def _complete_week_count(frame: pd.DataFrame) -> int:
    return sum(
        rows[TIME_COL].dt.normalize().nunique() == 7 and rows[TARGET].notna().all()
        for _, rows in frame.groupby("week_start")
    )


def _holiday_week(frame: pd.DataFrame) -> bool | None:
    if "holiday" not in frame:
        return None
    values = pd.to_numeric(frame["holiday"], errors="coerce")
    if values.notna().sum() == 0:
        return None
    values = values.fillna(0)
    return bool(values.gt(0).any())


def _make_md_report(result: dict[str, Any]) -> str:
    lines = ["# 유동인구 모델 백테스트", "", f"- 상태: **{result['status']}**"]
    if result.get("message"):
        lines.extend(["", result["message"]])
    if result.get("holdout_week"):
        lines.extend(["", f"- 최종 미사용 테스트 주: {result['holdout_week']}"])
        lines.extend(["- 테스트 자료는 행사 성과가 아니라 과거 유동인구 관측값입니다."])
    if result.get("metrics"):
        lines.extend(["", "## 최종 테스트 주 성능", "", "| 모델 | MAE | WAPE | 범위 포함률 | 평균 범위 폭 |", "|---|---:|---:|---:|---:|"])
        for name, metric in result["metrics"].items():
            lines.append(
                f"| {name} | {metric.get('mae', '—')} | {metric.get('wape', '—')} | "
                f"{metric.get('coverage_80', '—')} | {metric.get('mean_width', '—')} |"
            )
    rolling = result.get("rolling_validation", {})
    if rolling.get("summary"):
        lines.extend(["", "## 최종 테스트 주 이전 롤링 검증", "", "| 모델 | 검증 주 수 | 평균 MAE | 평균 WAPE | 구간 포함률 | 평균 범위 폭 |", "|---|---:|---:|---:|---:|---:|"])
        for name, metric in rolling["summary"].items():
            lines.append(
                f"| {name} | {metric.get('fold_count', '—')} | {metric.get('mean_mae', '—')} | "
                f"{metric.get('mean_wape', '—')} | {metric.get('mean_coverage_80', '—')} | {metric.get('mean_interval_width', '—')} |"
            )
        lines.append(f"\n롤링 검증에서 MAE가 가장 낮은 후보: **{rolling.get('lowest_rolling_mae_model') or '판정 불가'}**")
        lines.append("공휴일 주·평소 주 분리 요약은 `backtest_metrics.json`에서 확인합니다.")
    if result.get("holiday_metrics"):
        lines.extend(["", "## 공휴일 주와 평소 주", ""])
        for kind, metrics in result["holiday_metrics"].items():
            lines.append(f"- **{kind}**: {json.dumps(metrics, ensure_ascii=False)}")
    lines.extend([
        "",
        "## 해석 주의",
        "",
        "- MAE는 시간대별 예측과 실제값의 평균 차이입니다. WAPE는 전체 실제값 합계 대비 절대 오차 비율입니다.",
        "- 80% 범위는 과거 검증에서 실제값이 구간에 들어온 비율과 평균 폭을 함께 확인해야 합니다. 작은 표본에서는 보장으로 해석하지 않습니다.",
        "- 이 타깃은 장소 주변 유동인구입니다. 팝업 방문객·구매자·매출을 직접 예측한 결과가 아닙니다.",
        "- 주 단위로 날짜가 7일 모두 관측됐는지는 확인하지만, 기대한 모든 장소·시간대 슬롯이 수집됐는지는 운영시간/수집 계획표 없이는 판정할 수 없습니다.",
    ])
    return "\n".join(lines) + "\n"


def run_backtest(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        frame = _read_history(args.history)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"자료 확인 필요: {exc}", file=sys.stderr)
        return 2

    as_of = args.as_of or _local_today()
    holdout_start = _latest_complete_week(frame, as_of)
    result: dict[str, Any] = {
        "status": "insufficient_data",
        "as_of_kst": as_of.isoformat(),
        "source_file": str(args.history),
        "valid_rows": int(len(frame)),
        "data_quality": frame.attrs.get("quality", {}),
        "venue_count": int(frame["venue_id"].nunique()),
        "min_training_rows_for_ml": args.min_train_rows,
        "min_training_weeks_for_ml": args.min_train_weeks,
        "metrics": {},
        "holiday_metrics": {},
    }
    if holdout_start is None:
        result["message"] = "완료된 월~일 기간 중 실제 관측이 7일 모두 있는 테스트 주를 찾지 못했습니다. 테스트 기간과 실제값을 확인해 주세요."
    else:
        holdout = frame[frame["week_start"].eq(holdout_start)].copy()
        train = frame[frame["week_start"].lt(holdout_start)].copy()
        result["holdout_week"] = f"{holdout_start.date()} ~ {(holdout_start + pd.Timedelta(days=6)).date()}"
        train_weeks = _complete_week_count(train)
        result["training_rows"] = int(len(train))
        result["training_weeks"] = train_weeks
        result["rolling_validation"] = _rolling_cv(frame, holdout_start, args.min_train_rows, args.min_train_weeks)
        if train.empty:
            result["message"] = "최종 테스트 주보다 이전인 학습 자료가 없습니다."
        else:
            actual = holdout[TARGET].to_numpy(dtype=float)
            baseline = _fit_baseline(train, holdout)
            result["metrics"]["same-site weekday/hour median baseline"] = _metric(actual, baseline)
            result["status"] = "baseline_only"
            result["message"] = "기준선만 계산했습니다. ML 학습에는 최소 자료 기준이 필요하며, 기준선도 여러 과거 주 검증과 함께 해석해야 합니다."
            prediction_rows = holdout[["venue_id", TIME_COL, TARGET, "holiday"]].copy() if "holiday" in holdout else holdout[["venue_id", TIME_COL, TARGET]].copy()
            prediction_rows["baseline_median"] = baseline

            enough_ml_data = len(train) >= args.min_train_rows and train_weeks >= args.min_train_weeks
            if enough_ml_data:
                try:
                    poisson = _fit_poisson(train, holdout)
                    quantile_models, _ = _fit_quantiles(train)
                    interval = _apply_interval_margin(
                        _predict_quantiles(quantile_models, holdout),
                        result["rolling_validation"].get("quantile_interval_margin_80"),
                    )
                    result["metrics"]["Poisson gradient boosting"] = _metric(actual, poisson)
                    result["metrics"]["Quantile regression median"] = {
                        **_metric(actual, interval[:, 1]),
                        **_interval_metric(actual, interval[:, 0], interval[:, 2]),
                    }
                    prediction_rows["poisson_mean"] = poisson
                    prediction_rows["range_low_q10"] = interval[:, 0]
                    prediction_rows["range_median_q50"] = interval[:, 1]
                    prediction_rows["range_high_q90"] = interval[:, 2]
                    result["status"] = "evaluated"
                    result["message"] = "가장 최근의 완료 주를 학습에서 제외해 평가했습니다. 모델 선택은 마지막 테스트 성능이 아니라 그보다 과거의 롤링 검증 결과로 해야 합니다."
                except (ValueError, TypeError) as exc:
                    result["message"] = f"ML 계산을 완료하지 못해 기준선만 남겼습니다: {exc}"

            result["holiday_metrics"] = {"status": "available" if "holiday" in holdout else "holiday_column_missing"}
            if "holiday" in holdout:
                holiday_values = pd.to_numeric(holdout["holiday"], errors="coerce")
                if holiday_values.notna().sum() == 0:
                    result["holiday_metrics"]["status"] = "holiday_values_missing"
                else:
                    holiday_mask = holiday_values.fillna(0).gt(0)
                    for label, mask in (("holiday", holiday_mask), ("regular", ~holiday_mask)):
                        if mask.any():
                            result["holiday_metrics"][label] = {
                                "baseline": _metric(actual[mask.to_numpy()], baseline[mask.to_numpy()]),
                            }
                            if "poisson_mean" in prediction_rows:
                                result["holiday_metrics"][label]["poisson"] = _metric(actual[mask.to_numpy()], poisson[mask.to_numpy()])
                                result["holiday_metrics"][label]["quantile_median_and_range"] = {
                                    **_metric(actual[mask.to_numpy()], interval[mask.to_numpy(), 1]),
                                    **_interval_metric(actual[mask.to_numpy()], interval[mask.to_numpy(), 0], interval[mask.to_numpy(), 2]),
                                }
            prediction_rows.to_csv(out_dir / "holdout_predictions.csv", index=False, encoding="utf-8-sig")

    (out_dir / "backtest_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "backtest_report.md").write_text(_make_md_report(result), encoding="utf-8")
    print(f"백테스트 결과 저장: {out_dir.resolve()}")
    print(f"상태: {result['status']}")
    print(result.get("message", ""))
    return 0


def _share_profile(train: pd.DataFrame, scenario: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    combined = None
    if "age20_share" in train and "age30_share" in train:
        combined = train[["age20_share", "age30_share"]].sum(axis=1, min_count=2).clip(0, 1)

    if combined is None or combined.notna().sum() == 0:
        resident20 = scenario.get("resident_age20_share", pd.Series(np.nan, index=scenario.index))
        resident30 = scenario.get("resident_age30_share", pd.Series(np.nan, index=scenario.index))
        resident = (resident20 + resident30).where(resident20.notna() & resident30.notna())
        values = resident.clip(0, 1).to_numpy(dtype=float)
        quality = ["low_resident_proxy" if np.isfinite(value) else "missing_age_data" for value in values]
        return values, values, values, quality

    profile_data = train.copy()
    profile_data["target_age_share"] = combined
    group_values = profile_data["target_age_share"].dropna()
    global_q = group_values.quantile([0.1, 0.5, 0.9]).to_dict()
    by_site_time = profile_data.groupby(["venue_id", "weekday", "hour"])["target_age_share"].quantile([0.1, 0.5, 0.9]).unstack(-1)
    by_site = profile_data.groupby("venue_id")["target_age_share"].quantile([0.1, 0.5, 0.9]).unstack(-1)
    lows: list[float] = []
    meds: list[float] = []
    highs: list[float] = []
    quality: list[str] = []
    for row in scenario.itertuples(index=False):
        venue = getattr(row, "venue_id")
        weekday = getattr(row, "weekday")
        hour = getattr(row, "hour")
        values = by_site_time.loc[(venue, weekday, hour)] if (venue, weekday, hour) in by_site_time.index else None
        label = "venue_weekday_hour_history"
        if values is None and venue in by_site.index:
            values = by_site.loc[venue]
            label = "venue_history"
        if values is None:
            values = global_q
            label = "all_venue_history_low_confidence"
        lows.append(float(values[0.1]))
        meds.append(float(values[0.5]))
        highs.append(float(values[0.9]))
        quality.append(label)
    return np.clip(lows, 0, 1), np.clip(meds, 0, 1), np.clip(highs, 0, 1), quality


def run_forecast(args: argparse.Namespace) -> int:
    try:
        history = _read_history(args.history)
        scenarios = _prepare_scenarios(pd.read_csv(args.scenarios, encoding="utf-8-sig"))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        print(f"자료 확인 필요: {exc}", file=sys.stderr)
        return 2
    weeks = _complete_week_count(history)
    if scenarios.empty:
        print("예측할 장소·날짜·시간대 행이 없습니다.", file=sys.stderr)
        return 2
    if scenarios["forecast_at"].min() <= history[TIME_COL].max():
        print("미래 시나리오의 날짜·시간은 학습 자료의 마지막 관측 이후여야 합니다.", file=sys.stderr)
        return 2

    enough_ml_data = len(history) >= args.min_train_rows and weeks >= args.min_train_weeks
    holdout_start = _latest_complete_week(history, _local_today())
    validation = _rolling_cv(history, holdout_start, args.min_train_rows, args.min_train_weeks) if holdout_start is not None else {}
    selected_model = validation.get("lowest_rolling_mae_model")
    selected_summary = validation.get("summary", {}).get(selected_model, {})
    enough_selection_evidence = selected_summary.get("fold_count", 0) >= 3

    if enough_ml_data and enough_selection_evidence and selected_model in ("poisson", "quantile_regression"):
        quantile_models, _ = _fit_quantiles(history)
        q_bounds = _apply_interval_margin(
            _predict_quantiles(quantile_models, scenarios),
            validation.get("quantile_interval_margin_80"),
        )
        range_method = "quantile_regression_q10_q50_q90"
        range_basis = ["quantile_regression" for _ in range(len(scenarios))]
        if selected_model == "poisson":
            point = _fit_poisson(history, scenarios)
            selected_model_label = "Poisson gradient boosting (rolling CV lowest MAE)"
        else:
            point = q_bounds[:, 1]
            selected_model_label = "quantile regression Q50 (rolling CV lowest MAE)"
        # 선택한 점 예측을 범위에 포함해 중심 밖 예측이 생기는 것을 방지합니다.
        bounds = np.column_stack([np.minimum(q_bounds[:, 0], point), point, np.maximum(q_bounds[:, 2], point)])
    else:
        bounds, range_basis = _baseline_ranges(history, scenarios)
        point = _fit_baseline(history, scenarios)
        bounds = np.column_stack([np.minimum(bounds[:, 0], point), point, np.maximum(bounds[:, 2], point)])
        range_method = "historical_quantile_scenario_only"
        selected_model_label = "same-site weekday/hour median baseline"
        if not enough_ml_data:
            range_method += "_insufficient_ml_data"
        elif not enough_selection_evidence:
            range_method += "_insufficient_rolling_folds"
        elif selected_model == "baseline":
            range_method += "_baseline_won_rolling_cv"
    age_low, age_med, age_high, age_quality = _share_profile(history, scenarios)
    output = scenarios[["venue_id", "forecast_at"]].copy()
    output["footfall_low_estimate"] = bounds[:, 0]
    output["footfall_point_estimate"] = point
    output["footfall_high_estimate"] = bounds[:, 2]
    output["selected_model"] = selected_model_label
    output["range_method"] = range_method
    output["range_basis"] = range_basis
    output["range_calibration"] = (
        "rolling_out_of_fold_margin" if range_method.startswith("quantile_regression")
        and validation.get("quantile_interval_margin_80") is not None
        else "uncalibrated_quantiles_or_historical_scenario"
    )
    output["age20_30_share_low"] = age_low
    output["age20_30_share_median"] = age_med
    output["age20_30_share_high"] = age_high
    output["age20_30_count_low_scenario"] = bounds[:, 0] * age_low
    output["age20_30_count_median_scenario"] = point * age_med
    output["age20_30_count_high_scenario"] = bounds[:, 2] * age_high
    output["age_data_quality"] = age_quality
    output["weather_status"] = scenarios.get("weather_status", pd.Series("미판정", index=scenarios.index)).fillna("미판정").astype(str).to_numpy()

    normalized_weather = output["weather_status"].str.strip().str.lower()
    stop_tokens = ("stop", "exclude", "제외", "위험", "운영불가", "대안필요")
    output["outdoor_decision"] = np.where(
        normalized_weather.apply(lambda value: any(token in value for token in stop_tokens)),
        "제외/대안 검토",
        np.where(
            normalized_weather.isin(["caution", "주의", "경고"]),
            "주의 조건부 검토",
            np.where(normalized_weather.isin(["pass", "ok", "통과", "안전", "가능"]), "날씨 기준 통과", "날씨 상태 미확인"),
        ),
    )
    output["rankable"] = ~normalized_weather.apply(lambda value: any(token in value for token in stop_tokens))
    output = output.sort_values(
        ["rankable", "age20_30_count_median_scenario", "footfall_point_estimate"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    output.insert(0, "rank", np.arange(1, len(output) + 1))

    output_path: Path = args.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"후보 예측·비교 결과 저장: {output_path.resolve()}")
    print(f"점 예측 모델: {selected_model_label}")
    if range_method.startswith("quantile_regression"):
        print("범위 방식: 분위수 회귀 Q10/Q50/Q90; 중심이 다른 우승 모델이면 그 점 예측도 범위에 포함해 폭을 넓혔습니다.")
    else:
        print("범위 방식: 과거 관측 분위수 시나리오이며 통계적으로 검증된 예측구간은 아닙니다.")
    print("※ 인원 범위는 장소 주변 유동/체류 인구입니다. 팝업 방문객 수가 아니며, 20~30대 수는 과거 구성비 또는 낮은 신뢰도의 주민비율 대체 시나리오입니다.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    backtest = commands.add_parser("backtest", help="가장 최근 완전 주를 최종 테스트로 남겨 기준선과 회귀를 비교합니다.")
    backtest.add_argument("--history", type=Path, required=True, help="과거 실제 시간대별 유동인구 CSV")
    backtest.add_argument("--out-dir", type=Path, default=Path("outputs/modeling"))
    backtest.add_argument("--as-of", type=date.fromisoformat, help="KST 기준 날짜 YYYY-MM-DD; 기본값은 오늘")
    backtest.add_argument("--min-train-rows", type=int, default=MIN_MODEL_ROWS)
    backtest.add_argument("--min-train-weeks", type=int, default=MIN_MODEL_WEEKS)
    backtest.set_defaults(run=run_backtest)

    forecast = commands.add_parser("forecast", help="검증된 과거 관측으로 미래 후보 시간대의 유동인구 범위를 산출합니다.")
    forecast.add_argument("--history", type=Path, required=True)
    forecast.add_argument("--scenarios", type=Path, required=True, help="미래 후보 장소·시간·날씨·주변환경 CSV")
    forecast.add_argument("--out", type=Path, default=Path("outputs/modeling/candidate_forecasts.csv"))
    forecast.add_argument("--min-train-rows", type=int, default=MIN_MODEL_ROWS)
    forecast.add_argument("--min-train-weeks", type=int, default=MIN_MODEL_WEEKS)
    forecast.set_defaults(run=run_forecast)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
