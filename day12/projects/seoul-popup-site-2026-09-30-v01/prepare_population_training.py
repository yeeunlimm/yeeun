#!/usr/bin/env python3
"""Join Seoul population labels, archived hourly weather and Korean holiday features."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parent
POPULATION = ROOT / "data" / "processed" / "seoul_population_history.csv"
WEATHER = ROOT / "data" / "processed" / "seoul_weather_history.csv"
OUT = ROOT / "data" / "processed" / "seoul_population_training.csv"

# Irregular, officially designated public holidays within the training window.
SPECIAL_HOLIDAYS = {
    date(2023, 10, 2),  # temporary holiday
    date(2024, 10, 1),  # Armed Forces Day, temporary holiday
    date(2025, 1, 27),  # temporary holiday
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=POPULATION)
    parser.add_argument("--weather", type=Path, default=WEATHER)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    population = pd.read_csv(args.population, encoding="utf-8-sig", parse_dates=["observed_at"])
    weather = pd.read_csv(args.weather, encoding="utf-8-sig", parse_dates=["observed_at"])
    merged = population.merge(weather.drop(columns=["weather_source"], errors="ignore"), on="observed_at", how="left", validate="many_to_one")
    day_values = merged["observed_at"].dt.date
    year_set = range(merged["observed_at"].dt.year.min(), merged["observed_at"].dt.year.max() + 1)
    official = holidays.KR(years=year_set)
    holiday_days = set(official.keys()) | SPECIAL_HOLIDAYS
    merged["holiday"] = day_values.isin(holiday_days).astype(int)
    merged["holiday_eve"] = day_values.map(lambda day: int(day + timedelta(days=1) in holiday_days))
    merged["temporary_holiday"] = day_values.isin(SPECIAL_HOLIDAYS).astype(int)
    merged["month"] = merged["observed_at"].dt.month
    merged["weekday"] = merged["observed_at"].dt.weekday
    merged["hour"] = merged["observed_at"].dt.hour
    merged["is_weekend"] = merged["weekday"].isin([5, 6]).astype(int)
    missing_weather = int(merged["forecast_temperature_c"].isna().sum())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"학습 테이블 저장: {args.out.resolve()}")
    print(f"행 수: {len(merged):,} | 지역: {merged['venue_id'].nunique()}개 | 기간: {merged['observed_at'].min()}~{merged['observed_at'].max()}")
    print(f"날씨 결측: {missing_weather:,}행 | 공휴일 {merged['holiday'].sum():,}행 | 휴일 전날 {merged['holiday_eve'].sum():,}행 | 임시공휴일 {merged['temporary_holiday'].sum():,}행")
    if missing_weather:
        print("주의: 날씨 누락 행을 확인하세요.")
    print("※ 임시공휴일 비교는 학습 기간 중 지정 사례가 적어 일반적인 효과로 단정할 수 없습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
