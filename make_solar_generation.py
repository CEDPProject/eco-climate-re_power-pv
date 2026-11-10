# -*- coding: utf-8 -*-

"""
==================================================
시도별 태양광 잠재발전량 계산 (2단계)

입력: output/solar_grid/*.npz
출력: output/solar_generation/solar_generation_sido.csv
==================================================
"""

import os
import numpy as np
import pandas as pd
import shapefile
from pyproj import Transformer
from matplotlib.path import Path


# ==================================================
# PATH 설정
# ==================================================

BASE_DIR = "/data/jyy/S82_2025/KIML_monthly"

GRID_DIR = os.path.join(
    BASE_DIR,
    "output",
    "solar_grid"
)

SHP = os.path.join(
    BASE_DIR,
    "shp",
    "sido",
    "bnd_sido_00_2025_2Q.shp"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "solar_generation"
)

os.makedirs(OUT_DIR, exist_ok=True)


# ==================================================
# 시도별 태양광 설비용량 (kW)
# ==================================================

capacity = {
    '서울': 247791,
    '부산': 317737,
    '대구': 420080,
    '인천': 252181,
    '광주': 364304,
    '대전': 131387,
    '울산': 165315,
    '세종': 107164,
    '경기': 2130246,
    '강원': 1979034,
    '충북': 1636344,
    '충남': 3498732,
    '전북': 4607206,
    '전남': 5903208,
    '경북': 3561773,
    '경남': 2056529,
    '제주': 654314
}


# ==================================================
# 시도 SHP 읽기 및 좌표 변환 (EPSG:5179 -> EPSG:4326)
# ==================================================

sf = shapefile.Reader(SHP, encoding="utf-8")
fields = [x[0] for x in sf.fields[1:]]
idx = fields.index("SIDO_CD")

sido_map = {
    '11': '서울', '21': '부산', '22': '대구', '23': '인천', '24': '광주',
    '25': '대전', '26': '울산', '29': '세종', '31': '경기', '32': '강원',
    '33': '충북', '34': '충남', '35': '전북', '36': '전남', '37': '경북',
    '38': '경남', '39': '제주'
}

# Transformer 객체 1회 선언 (속도 향상)
transformer = Transformer.from_crs("EPSG:5179", "EPSG:4326", always_xy=True)

polygons = []

for shape, rec in zip(sf.shapes(), sf.records()):
    code = str(rec[idx])
    sido = sido_map.get(code)

    if sido:
        pts = []
        for x, y in shape.points:
            lon, lat = transformer.transform(x, y)
            pts.append([lon, lat])

        polygons.append((sido, np.array(pts)))

print(f"✅ SIDO polygon 로드 완료: {len(polygons)}개 시도 구역")


# ==================================================
# MAIN: 월별 npz 격자 데이터 읽기 및 잠재발전량 계산
# ==================================================

results = []

for fname in sorted(os.listdir(GRID_DIR)):
    if not fname.endswith(".npz"):
        continue

    print("PROCESS:", fname)
    month = fname.replace(".npz", "")

    data = np.load(os.path.join(GRID_DIR, fname))
    lat = data["lat"]
    lon = data["lon"]
    density = data["density"]

    # 유효 데이터(NaN 제외) 추출
    valid = ~np.isnan(density)
    points = np.column_stack((lon[valid], lat[valid]))
    values = density[valid]

    print(f"  valid grid: {len(values)}")

    # 시도별 Spatial Masking & 발전량 집계
    for sido, poly in polygons:
        # Bounding Box 1차 필터링
        minx, miny = poly.min(axis=0)
        maxx, maxy = poly.max(axis=0)

        idx_box = (
            (points[:, 0] >= minx) & (points[:, 0] <= maxx) &
            (points[:, 1] >= miny) & (points[:, 1] <= maxy)
        )

        sub_points = points[idx_box]
        sub_values = values[idx_box]

        if len(sub_values) == 0:
            continue

        # Polygon 내격자 2차 Precise 마스킹
        polygon_path = Path(poly)
        mask = polygon_path.contains_points(sub_points)

        if mask.sum() == 0:
            continue

        # 1) 시도 일평균 에너지 밀도 (kWh/m2/day)
        mean_density = float(np.mean(sub_values[mask]))

        # 2) 일평균 잠재발전량 산출 (MWh/day)
        # 공식: (에너지 밀도 * 설비용량 * 성능계수 0.75) / 1000
        generation = mean_density * capacity[sido] * 0.75 / 1000.0

        results.append({
            "month": month,
            "sido": sido,
            "energy_density_kWh_m2_day": round(mean_density, 4),
            "capacity_kW": capacity[sido],
            "generation_MWh_day": round(generation, 3)
        })


# ==================================================
# SAVE CSV
# ==================================================

df = pd.DataFrame(results)

outfile = os.path.join(OUT_DIR, "solar_generation_sido.csv")
df.to_csv(outfile, index=False, encoding="utf-8-sig")

print("\n--------------------------------------------------")
print("✅ 태양광 시도별 잠재발전량 집계 완료!")
print(f"📁 저장 위치: {outfile}")
print("--------------------------------------------------")
print(df.head(10))