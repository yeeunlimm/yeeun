"""Download and summarize one fully published week of Seoul 250m population data.

The output describes estimated people present in administrative areas, not
pedestrian counts or popup visitors. Requires requests (already in requirements).
"""
from __future__ import annotations

import csv
import io
import re
import statistics
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PAGE = "https://data.seoul.go.kr/dataList/OA-22784/F/1/datasetView.do"
DOWNLOAD = "https://datafile.seoul.go.kr/bigfile/iot/inf/nio_download.do?useCache=false"

# Candidate districts came from the Sweetspot trend page; these are broad
# administrative-dong proxies, not confirmed event addresses.
AREAS = {
    "Seongsu": {"11200650", "11200660", "11200670", "11200690"},
    "Hongdae": {"11440660"},
    "Yongsan-Ipark": {"11170625"},  # Hangangno-dong
    "Myeongdong": {"11140550"},
    "Jamsil": {"11710680", "11710710"},  # Jamsil 3 and 6
}
AGE_20_39_COLS = (8, 9, 10, 11, 22, 23, 24, 25)


def list_published_files(session: requests.Session) -> dict[date, str]:
    page = session.get(PAGE, timeout=60)
    page.raise_for_status()
    files: dict[date, str] = {}
    for row in re.findall(r'<tr id="fileTr_\d+">.*?</tr>', page.text, re.S):
        name = re.search(r'title="(250_LOCAL_RESD_(\d{8})\.zip)"', row)
        seq = re.search(r"downloadFile\('(\d+)'\)", row)
        if name and seq:
            files[date.fromisoformat(f"{name.group(2)[:4]}-{name.group(2)[4:6]}-{name.group(2)[6:]}")] = seq.group(1)
    if not files:
        raise RuntimeError("서울시 파일 목록을 읽지 못했습니다.")
    return files


def latest_complete_week(files: dict[date, str]) -> list[date]:
    newest = max(files)
    sunday = newest - timedelta(days=(newest.weekday() + 1) % 7)
    while True:
        monday = sunday - timedelta(days=6)
        week = [monday + timedelta(days=i) for i in range(7)]
        if all(day in files for day in week):
            return week
        sunday -= timedelta(days=7)


def download_week(week: list[date], files: dict[date, str]) -> list[Path]:
    RAW.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    paths = []
    for day in week:
        path = RAW / f"seoul_presence_{day:%Y%m%d}.zip"
        if not path.exists() or not zipfile.is_zipfile(path):
            seq = files[day]
            response = session.post(
                DOWNLOAD,
                params={"run": day.isoformat()},
                data={"infId": "OA-22784", "seq": seq, "infSeq": "1"},
                timeout=180,
            )
            response.raise_for_status()
            if not zipfile.is_zipfile(io.BytesIO(response.content)):
                raise RuntimeError(f"{day} 파일을 ZIP으로 받지 못했습니다.")
            path.write_bytes(response.content)
        paths.append(path)
    return paths


def summarize(paths: list[Path], week: list[date]) -> Path:
    # key -> [estimated present population, estimated 20-39 population]
    sums: dict[tuple[str, str, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for path in paths:
        with zipfile.ZipFile(path) as archive:
            with archive.open(archive.namelist()[0]) as binary:
                rows = csv.reader(io.TextIOWrapper(binary, encoding="cp949", newline=""))
                next(rows, None)
                for row in rows:
                    if len(row) < 33:
                        continue
                    dong = row[2].strip()
                    area = next((name for name, codes in AREAS.items() if dong in codes), None)
                    if area is None:
                        continue
                    try:
                        total = float(row[4])
                        age20_39 = sum(float(row[i]) for i in AGE_20_39_COLS if row[i].strip() != "*")
                        hour = int(row[1])
                    except (ValueError, IndexError):
                        continue
                    entry = sums[(area, row[0].strip(), hour)]
                    entry[0] += total
                    entry[1] += age20_39

    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / f"seoul_presence_week_{week[0]:%Y%m%d}_{week[-1]:%Y%m%d}.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["area", "date", "hour", "estimated_present_total", "estimated_present_age20_39"])
        for (area, day, hour), (total, cohort) in sorted(sums.items()):
            writer.writerow([area, day, hour, round(total, 3), round(cohort, 3)])
    return output


def main() -> None:
    session = requests.Session()
    files = list_published_files(session)
    week = latest_complete_week(files)
    paths = download_week(week, files)
    output = summarize(paths, week)
    data: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    with output.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            data[row["area"]].append((int(row["hour"]), float(row["estimated_present_total"]), float(row["estimated_present_age20_39"])))
    print(f"기간: {week[0]}~{week[-1]} (완료된 주, 250m 체류인구)")
    print("지역 | 시간당 평균 추정 체류인구 | 그중 20~39세 | 평균 최고 시간대")
    for area in AREAS:
        rows = data.get(area, [])
        if not rows:
            print(f"{area} | 자료 없음")
            continue
        by_hour = [(hour, statistics.mean(t for h, t, _ in rows if h == hour), statistics.mean(a for h, _, a in rows if h == hour)) for hour in range(24)]
        mean_total = statistics.mean(total for _, total, _ in by_hour)
        mean_cohort = statistics.mean(cohort for _, _, cohort in by_hour)
        peak = max(by_hour, key=lambda row: row[1])
        print(f"{area} | {mean_total:,.0f} | {mean_cohort:,.0f} | {peak[0]}시 ({peak[1]:,.0f})")
    print(f"요약 파일: {output}")
    print("주의: 체류인구는 통신자료 기반 추정치이며 통행량, 팝업 방문객 수 또는 일 누적 방문 수가 아닙니다.")


if __name__ == "__main__":
    main()
