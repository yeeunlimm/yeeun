#!/usr/bin/env python3
"""Download hourly historical forecast-model weather for Seoul (no API key required)."""
from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "processed" / "seoul_weather_history.csv"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 9, 21))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 20))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    response = requests.get(URL, params={
        "latitude": 37.5665,
        "longitude": 126.9780,
        "start_date": args.start.isoformat(),
        "end_date": args.end.isoformat(),
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "wind_speed_unit": "ms",
        "timezone": "Asia/Seoul",
    }, timeout=180)
    response.raise_for_status()
    payload = response.json()
    hourly = payload["hourly"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observed_at", "forecast_temperature_c", "forecast_precipitation_mm", "forecast_wind_speed_mps", "weather_source"])
        for i, stamp in enumerate(hourly["time"]):
            writer.writerow([
                f"{stamp}:00",
                hourly["temperature_2m"][i],
                hourly["precipitation"][i],
                hourly["wind_speed_10m"][i],
                "Open-Meteo historical forecast model archive",
            ])
    print(f"서울 시간별 날씨 저장: {args.out.resolve()}")
    print(f"행 수: {len(hourly['time']):,} | 기간: {hourly['time'][0]}~{hourly['time'][-1]}")
    print("※ 기상청 관측값이 아닌 과거 예보모델 아카이브 값입니다. 미래 예보와 유사한 입력 형식의 과거 자료로 학습에 사용합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
