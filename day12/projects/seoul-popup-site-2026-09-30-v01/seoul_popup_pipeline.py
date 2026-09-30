#!/usr/bin/env python3
"""Collect and inspect public inputs for the Seoul pop-up location workflow."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, unquote

import requests

SEOUL_BASE = "https://openapi.seoul.go.kr:8088"
KMA_SHORT_BASE = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
KMA_MID_BASE = "https://apis.data.go.kr/1360000/MidFcstInfoService"
TIMEOUT_SECONDS = 25
POPUP_TERMS = ("팝업", "popup", "pop-up")


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Load simple KEY=VALUE entries without overriding existing environment variables."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        name, value = entry.split("=", 1)
        name, value = name.strip(), value.strip()
        if value and value[0:1] in ("'", '"') and value[-1:] == value[0:1]:
            value = value[1:-1]
        if name:
            os.environ.setdefault(name, value)


def _service_key(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(
            f"환경변수 {env_name}가 비어 있습니다. API 키를 코드에 적지 말고 "
            "PowerShell 환경변수로 설정해 주세요."
        )
    # 공공데이터포털에서 복사한 URL 인코딩 키를 requests가 다시 인코딩하지 않도록
    # 한 번 디코딩한 뒤 requests에 전달합니다.
    return unquote(value)


def _request_json(label: str, url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        # requests 예외 문자열에는 인증키를 포함한 전체 URL이 들어갈 수 있어 출력하지 않습니다.
        suffix = f" HTTP {status}" if status else ""
        raise RuntimeError(f"{label} 요청 실패.{suffix} 네트워크와 서비스 신청 상태를 확인해 주세요.") from None
    except ValueError:
        raise RuntimeError(f"{label} 응답을 JSON으로 읽지 못했습니다.") from None

    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} 응답의 최상위 형식이 JSON 객체가 아닙니다.")
    # 공공데이터포털 API는 HTTP 200이어도 본문에 오류 코드를 반환할 수 있습니다.
    header = payload.get("response", {}).get("header", {}) if isinstance(payload.get("response"), dict) else {}
    result_code = str(header.get("resultCode", "00"))
    if result_code not in ("00", "0"):
        raise RuntimeError(f"{label} 서비스 오류(resultCode={result_code}). API 신청/승인과 요청 조건을 확인해 주세요.")
    seoul_result = payload.get("RESULT")
    if isinstance(seoul_result, dict) and str(seoul_result.get("CODE", "")).upper() not in ("INFO-000", ""):
        raise RuntimeError(f"{label} 서비스 오류({seoul_result.get('CODE')}). API 신청/승인과 장소 이름을 확인해 주세요.")
    return payload


def _save_response(payload: dict[str, Any], output_dir: Path, stem: str, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output_dir / f"{stem}-{stamp}.json"
    document = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": metadata,
        "response": payload,
    }
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def collect_seoul_citydata(area_name: str) -> dict[str, Any]:
    key = quote(_service_key("SEOUL_DATA_API_KEY"), safe="")
    area = quote(area_name.strip(), safe="")
    if not area:
        raise ValueError("서울시 핫스팟 이름을 입력해 주세요.")
    # 서울시 API는 인증키가 경로에 포함되므로 URL이나 요청 객체를 로그로 출력하지 않습니다.
    url = f"{SEOUL_BASE}/{key}/json/citydata/1/5/{area}"
    payload = _request_json("서울시 실시간 도시데이터", url)
    return payload


def collect_kma_mid_land(tm_fc: str, reg_id: str = "11B00000") -> dict[str, Any]:
    key = _service_key("KMA_SERVICE_KEY")
    params = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 10,
        "dataType": "JSON",
        "regId": reg_id,
        "tmFc": tm_fc,
    }
    return _request_json("기상청 중기육상예보", f"{KMA_MID_BASE}/getMidLandFcst", params=params)


def collect_kma_short(base_date: str, base_time: str, nx: int, ny: int) -> dict[str, Any]:
    key = _service_key("KMA_SERVICE_KEY")
    params = {
        "serviceKey": key,
        "pageNo": 1,
        "numOfRows": 1000,
        "dataType": "JSON",
        "base_date": base_date,
        "base_time": base_time,
        "nx": nx,
        "ny": ny,
    }
    return _request_json("기상청 단기예보", f"{KMA_SHORT_BASE}/getVilageFcst", params=params)


def _decode_csv(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig"), "utf-8-sig"
    try:
        return raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        # 제공된 서울시 CSV는 CP949 계열입니다. 비정상 바이트는 대체 문자로 남겨
        # 원문 링크 검수 대상으로 표시하고 분석을 멈추지 않습니다.
        return raw.decode("cp949", errors="replace"), "cp949 (errors replaced)"


def _unique_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    unique: list[str] = []
    for header in headers:
        base = (header or "").strip()
        counts[base] = counts.get(base, 0) + 1
        unique.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return unique


def _parse_date(value: str) -> date | None:
    candidate = (value or "").strip()[:10]
    try:
        return date.fromisoformat(candidate)
    except ValueError:
        return None


def _event_period(row: dict[str, str]) -> tuple[date | None, date | None]:
    start = _parse_date(row.get("시작일", ""))
    end = _parse_date(row.get("종료일", ""))
    if start and end:
        return start, end
    raw = row.get("날짜", "")
    pieces = raw.split("~", 1)
    if len(pieces) == 2:
        return _parse_date(pieces[0]), _parse_date(pieces[1])
    return None, None


def _is_popup(row: dict[str, str]) -> bool:
    text = f"{row.get('공연/행사명', '')} {row.get('장소', '')}".casefold()
    return any(term in text for term in POPUP_TERMS)


def inspect_event_csv(path: Path, start: date | None = None, end: date | None = None) -> dict[str, Any]:
    text, encoding = _decode_csv(path)
    import io

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        headers = _unique_headers(next(reader))
    except StopIteration:
        raise ValueError("CSV 파일이 비어 있습니다.") from None

    records: list[dict[str, str]] = []
    width_mismatches = 0
    for raw_row in reader:
        if len(raw_row) != len(headers):
            width_mismatches += 1
            raw_row = (raw_row + [""] * len(headers))[: len(headers)]
        records.append(dict(zip(headers, raw_row)))

    popup_records = [row for row in records if _is_popup(row)]
    popup_overlapping: list[dict[str, Any]] = []
    invalid_dates = 0
    reversed_periods = 0
    implausible_years = 0
    missing_longitude = 0
    missing_latitude = 0

    for row in records:
        event_start, event_end = _event_period(row)
        if not event_start or not event_end:
            invalid_dates += 1
        else:
            if event_end < event_start:
                reversed_periods += 1
            if event_start.year < 2014 or event_start.year > 2027 or event_end.year < 2014 or event_end.year > 2027:
                implausible_years += 1
            if _is_popup(row) and (start is None or event_end >= start) and (end is None or event_start <= end):
                popup_overlapping.append(
                    {
                        "district": row.get("자치구", ""),
                        "title": row.get("공연/행사명", ""),
                        "period": row.get("날짜", ""),
                        "venue": row.get("장소", ""),
                        "detail_url": row.get("문화포털상세URL", ""),
                    }
                )
        missing_longitude += not bool(row.get("경도(X좌표)", "").strip())
        missing_latitude += not bool(row.get("위도(Y좌표)", "").strip())

    return {
        "source_file": str(path),
        "encoding": encoding,
        "row_count": len(records),
        "column_count": len(headers),
        "columns": headers,
        "row_width_mismatches": width_mismatches,
        "popup_keyword_rows": len(popup_records),
        "popup_rows_overlapping_window": popup_overlapping,
        "date_parse_failures": invalid_dates,
        "end_before_start": reversed_periods,
        "implausible_years_outside_2014_2027": implausible_years,
        "missing_longitude": missing_longitude,
        "missing_latitude": missing_latitude,
        "filter_window": {"start": start.isoformat() if start else None, "end": end.isoformat() if end else None},
    }


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _date_arg(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("날짜는 YYYY-MM-DD 형식으로 입력해 주세요.") from None


def main(argv: Iterable[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect-events", help="서울 문화행사 CSV에서 팝업/기간/좌표를 점검합니다.")
    inspect.add_argument("--csv", type=Path, required=True)
    inspect.add_argument("--start", type=_date_arg)
    inspect.add_argument("--end", type=_date_arg)
    inspect.add_argument("--out", type=Path, default=Path("data/processed/culture_event_profile.json"))

    seoul = commands.add_parser("collect-seoul", help="서울시 핫스팟 실시간 도시데이터를 저장합니다.")
    seoul.add_argument("--area", required=True, help="서울시 API가 지원하는 핫스팟 이름")
    seoul.add_argument("--out-dir", type=Path, default=Path("data/raw/seoul"))

    mid = commands.add_parser("collect-mid-weather", help="서울/경기 중기육상예보를 저장합니다.")
    mid.add_argument("--tm-fc", required=True, help="최근 발표시각 YYYYMMDD0600 또는 YYYYMMDD1800")
    mid.add_argument("--reg-id", default="11B00000", help="서울·인천·경기 중기육상예보 구역코드")
    mid.add_argument("--out-dir", type=Path, default=Path("data/raw/kma"))

    short = commands.add_parser("collect-short-weather", help="기상청 동네 단기예보를 저장합니다.")
    short.add_argument("--base-date", required=True, help="예보 발표일 YYYYMMDD")
    short.add_argument("--base-time", required=True, help="예보 발표시각 HHMM")
    short.add_argument("--nx", required=True, type=int, help="후보 장소의 기상 격자 X")
    short.add_argument("--ny", required=True, type=int, help="후보 장소의 기상 격자 Y")
    short.add_argument("--out-dir", type=Path, default=Path("data/raw/kma"))

    args = parser.parse_args(argv)
    try:
        if args.command == "inspect-events":
            result = inspect_event_csv(args.csv, args.start, args.end)
            _write_json(result, args.out)
            print(f"행사 CSV 점검 요약 저장: {args.out}")
            print(json.dumps({k: v for k, v in result.items() if k != "columns"}, ensure_ascii=False, indent=2))
        elif args.command == "collect-seoul":
            response = collect_seoul_citydata(args.area)
            path = _save_response(response, args.out_dir, "seoul-citydata", {"area": args.area})
            print(f"서울시 응답 저장: {path}")
        elif args.command == "collect-mid-weather":
            response = collect_kma_mid_land(args.tm_fc, args.reg_id)
            path = _save_response(response, args.out_dir, "kma-mid-land", {"tmFc": args.tm_fc, "regId": args.reg_id})
            print(f"기상청 중기예보 응답 저장: {path}")
        elif args.command == "collect-short-weather":
            response = collect_kma_short(args.base_date, args.base_time, args.nx, args.ny)
            path = _save_response(
                response,
                args.out_dir,
                "kma-short",
                {"base_date": args.base_date, "base_time": args.base_time, "nx": args.nx, "ny": args.ny},
            )
            print(f"기상청 단기예보 응답 저장: {path}")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
