#!/usr/bin/env python3
"""Collect and aggregate Seoul's monthly 250m population archives for candidate areas."""
from __future__ import annotations

import argparse
import csv
import io
import re
import zipfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "seoul_history"
OUT = ROOT / "data" / "processed" / "seoul_population_history.csv"
PAGE = "https://data.seoul.go.kr/dataList/OA-22784/F/1/datasetView.do"
DOWNLOAD = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false"
AREAS = {
    "Seongsu": {"11200650", "11200660", "11200670", "11200690"},
    "Hongdae": {"11440660"},
    "Yongsan-Ipark": {"11170625"},
    "Myeongdong": {"11140550"},
    "Jamsil": {"11710680", "11710710"},
}
AGE_20_39_COLS = (8, 9, 10, 11, 22, 23, 24, 25)


def published_archives(session: requests.Session) -> dict[date, tuple[str, str]]:
    response = session.get(PAGE, timeout=60)
    response.raise_for_status()
    result: dict[date, tuple[str, str]] = {}
    rows = re.findall(r'<tr id="fileTr_\d+">.*?</tr>', response.text, re.S)
    for row in rows:
        match = re.search(r'title="250_LOCAL_RESD_(\d{6}|\d{8})\.zip"', row)
        seq = re.search(r"downloadFile\('([^']+)'\)", row)
        if not match or not seq:
            continue
        token = match.group(1)
        if len(token) == 8:
            day = date(int(token[:4]), int(token[4:6]), int(token[6:8]))
            result[day] = (seq.group(1), "day")
        else:
            month = date(int(token[:4]), int(token[4:6]), 1)
            result[month] = (seq.group(1), "month")
    if not result:
        raise RuntimeError("서울시 자료 목록을 읽지 못했습니다.")
    return result


def month_starts(start: date, end: date) -> list[date]:
    cursor = date(start.year, start.month, 1)
    months = []
    while cursor <= end:
        months.append(cursor)
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
    return months


def archive_path(day: date, kind: str) -> Path:
    label = day.strftime("%Y%m" if kind == "month" else "%Y%m%d")
    return RAW / f"250_LOCAL_RESD_{label}.zip"


def download_archive(session: requests.Session, day: date, spec: tuple[str, str]) -> Path:
    seq, kind = spec
    path = archive_path(day, kind)
    if path.is_file() and zipfile.is_zipfile(path):
        return path
    RAW.mkdir(parents=True, exist_ok=True)
    label = day.isoformat() if kind == "day" else day.strftime("%Y-%m")
    with session.post(DOWNLOAD, params={"run": label}, data={"infId": "OA-22784", "seq": seq, "infSeq": "1"}, timeout=(30, 900), stream=True) as response:
        response.raise_for_status()
        temp = path.with_suffix(".part")
        with temp.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
        if not zipfile.is_zipfile(temp):
            temp.unlink(missing_ok=True)
            raise RuntimeError(f"다운로드 파일이 ZIP 형식이 아닙니다: {label}")
        temp.replace(path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2023, 9, 21))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 9, 20))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end는 --start 이후여야 합니다.")

    session = requests.Session()
    index = published_archives(session)
    selected: list[tuple[date, str, tuple[str, str]]] = []
    for month in month_starts(args.start, args.end):
        if month in index and index[month][1] == "month":
            selected.append((month, "month", index[month]))
        else:
            # Recent month archives may not be published yet; use available daily files.
            for day, spec in sorted(index.items()):
                if spec[1] == "day" and day.year == month.year and day.month == month.month and args.start <= day <= args.end:
                    selected.append((day, "day", spec))
    selected = list({(day, kind): (day, kind, spec) for day, kind, spec in selected}.values())
    expected_months = len(month_starts(args.start, args.end))
    if not selected:
        raise RuntimeError("요청한 기간에 내려받을 수 있는 파일이 없습니다.")

    # Aggregate by area and hour; raw 250m cells are summed within each predefined proxy area.
    aggregates: list[pd.DataFrame] = []
    files = sorted(selected, key=lambda item: item[0])
    print(f"수집 대상: {args.start}~{args.end}, 월 묶음 {expected_months}개에 해당하는 파일 {len(files)}개", flush=True)
    for number, (archive_date, kind, spec) in enumerate(files, 1):
        path = download_archive(session, archive_date, spec)
        label = archive_date.strftime("%Y-%m" if kind == "month" else "%Y-%m-%d")
        print(f"[{number}/{len(files)}] 파일 확보 {label} ({path.stat().st_size / 1_000_000:.1f} MB)", flush=True)
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                with archive.open(name) as binary:
                    chunks = pd.read_csv(binary, encoding="cp949", usecols=[0, 1, 2, 4, *AGE_20_39_COLS], dtype=str, chunksize=500_000)
                    for frame in chunks:
                        frame.columns = ["date", "hour", "dong", "total", "age1", "age2", "age3", "age4", "age5", "age6", "age7", "age8"]
                        frame["date"] = pd.to_datetime(frame["date"].str[:8], format="%Y%m%d", errors="coerce").dt.date
                        frame = frame[frame["date"].between(args.start, args.end)]
                        frame["area"] = frame["dong"].str.strip().map({code: area for area, codes in AREAS.items() for code in codes})
                        frame = frame[frame["area"].notna()]
                        if frame.empty:
                            continue
                        numeric = ["total", "age1", "age2", "age3", "age4", "age5", "age6", "age7", "age8"]
                        frame[numeric] = frame[numeric].replace("*", "0").apply(pd.to_numeric, errors="coerce").fillna(0)
                        frame["population_20_39"] = frame[["age1", "age2", "age3", "age4", "age5", "age6", "age7", "age8"]].sum(axis=1)
                        grouped = frame.groupby(["area", "date", "hour"], as_index=False)[["total", "population_20_39"]].sum()
                        aggregates.append(grouped)
        print(f"[{number}/{len(files)}] 집계 완료 {label}", flush=True)

    if not aggregates:
        raise RuntimeError("후보 지역에 해당하는 유효한 생활인구 행을 찾지 못했습니다.")
    totals = pd.concat(aggregates, ignore_index=True).groupby(["area", "date", "hour"], as_index=False)[["total", "population_20_39"]].sum()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["venue_id", "observed_at", "footfall_total", "population_20_39", "age20_39_share", "source_unit"])
        for row in totals.sort_values(["area", "date", "hour"]).itertuples(index=False):
            area, day, hour, total, cohort = row
            writer.writerow([area, f"{day} {int(hour):02d}:00:00", round(total, 3), round(cohort, 3), round(cohort / total, 6) if total else 0, "administrative_dong_250m_grid_sum"])
    print(f"학습용 시간대 CSV: {args.out.resolve()}", flush=True)
    print(f"행 수: {len(totals):,} | 지역 수: {totals['area'].nunique()} | 기간: {args.start}~{args.end}", flush=True)
    print("주의: 이는 행정동 단위 250m 생활인구 격자 합계의 대리 지표이며 보행량이나 팝업 방문객 수가 아닙니다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
