from pathlib import Path
import re

root = Path(__file__).resolve().parent
report = root / "report.md"
text = report.read_text(encoding="utf-8")
required = [
    "2026년 10월 10일(토)",
    "성수",
    "팝업 방문객 1만 명",
    "서울시",
]
missing = [item for item in required if item not in text]
images = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", text)
missing_images = [path for path in images if not (root / path).is_file()]
if missing or missing_images:
    print(f"FAIL: missing text={missing}; missing images={missing_images}")
    raise SystemExit(1)
print(f"PASS: report exists; {len(images)} linked charts exist; date and caveats present")
