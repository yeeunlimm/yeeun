"""Build report figures from the reviewed weekly presence summary CSV."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
source = ROOT / "data/processed/seoul_presence_week_20260914_20260920.csv"
out_dir = ROOT / "data/processed/figures"
out_dir.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(source.open(encoding="utf-8-sig", newline="")))
areas = ["Seongsu", "Jamsil", "Hongdae", "Yongsan-Ipark", "Myeongdong"]
labels = ["성수", "잠실", "홍대", "용산 아이파크몰 권역", "명동"]
hourly = defaultdict(lambda: defaultdict(list))
for row in rows:
    area, hour = row["area"], int(row["hour"])
    hourly[area][hour].append((float(row["estimated_present_total"]), float(row["estimated_present_age20_39"])))

total = [np.mean([t for h in hourly[a].values() for t, _ in h]) for a in areas]
age2039 = [np.mean([c for h in hourly[a].values() for _, c in h]) for a in areas]

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1.2, 1]})
fig.patch.set_facecolor("white")
for ax in axes:
    ax.set_facecolor("white")
    ax.grid(axis="x", color="#E7EBF0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#CBD2DA")
    ax.tick_params(axis="y", length=0, labelsize=10, pad=7)
    ax.tick_params(axis="x", colors="#596273", labelsize=9)

y = np.arange(len(areas))
axes[0].barh(y, np.array(total) / 10000, color="#1B4965", height=0.62)
axes[0].set_yticks(y, labels)
axes[0].invert_yaxis()
axes[0].set_xlabel("시간당 평균 추정 체류인구 (만 명)", color="#596273", fontsize=9)
axes[0].set_title("권역 전체", loc="left", fontsize=13, fontweight="bold", color="#17212B", pad=14)
for yy, val in zip(y, total):
    axes[0].text(val / 10000 + .08, yy, f"{val/10000:.1f}만", va="center", fontsize=9, color="#17212B")

axes[1].barh(y, np.array(age2039) / 10000, color="#E07A5F", height=0.62)
axes[1].set_yticks(y, labels)
axes[1].invert_yaxis()
axes[1].set_xlabel("시간당 평균 추정 체류인구 (만 명)", color="#596273", fontsize=9)
axes[1].set_title("20~39세", loc="left", fontsize=13, fontweight="bold", color="#17212B", pad=14)
for yy, val in zip(y, age2039):
    axes[1].text(val / 10000 + .05, yy, f"{val/10000:.1f}만", va="center", fontsize=9, color="#17212B")
fig.suptitle("서울 후보 권역별 시간당 평균 추정 체류인구", x=.06, y=1.02, ha="left", fontsize=16, fontweight="bold", color="#17212B")
fig.text(.06, -.015, "2026.09.14–20 · 서울시 250m 생활인구 · 행정동 합산 · 방문객/통행량 아님", fontsize=9, color="#596273")
fig.tight_layout()
fig.savefig(out_dir / "area_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)

matrix = np.array([[np.mean([c for t, c in hourly[a][h]]) for h in range(24)] for a in areas]) / 10000
fig, ax = plt.subplots(figsize=(12, 4.4))
im = ax.imshow(matrix, cmap="YlOrBr", aspect="auto", interpolation="nearest")
ax.set_yticks(np.arange(len(areas)), labels)
ax.set_xticks(np.arange(0, 24, 2), [f"{h}시" for h in range(0, 24, 2)])
ax.set_title("시간대별 20~39세 추정 체류인구", loc="left", fontsize=15, fontweight="bold", color="#17212B", pad=14)
ax.set_xlabel("시간대", color="#596273")
ax.tick_params(axis="both", length=0, labelsize=9, pad=7)
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=.015)
cb.set_label("시간당 평균 추정 인원 (만 명)", color="#596273", fontsize=9)
cb.outline.set_visible(False)
fig.text(.08, -.005, "각 시간대는 7일 평균. 연령별 소수 인원(*)은 원자료에서 비식별 처리되어 연령 규모가 낮게 잡힐 수 있습니다.", fontsize=9, color="#596273")
fig.tight_layout()
fig.savefig(out_dir / "age2039_hourly_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"created: {out_dir}")
