# %%
import os
import gc
import pandas as pd
import geopandas as gpd
import fiona

# ============================
# 配置（已按你截图修好）
# ============================
BUSINESS_FILE = "Phi_Restaurant.csv"     # 你的商家表
ID_COL  = "business_id"
LAT_COL = "lat"
LON_COL = "lon"

RADII = [100, 200, 500]        # 米

OSM_GPKGS = {
    # 点
    "pois":        "Philadelphia_gis_osm_pois_free_1.gpkg",
    "places":     "Philadelphia_gis_osm_places_free_1.gpkg",
    "traffic":    "Philadelphia_gis_osm_traffic_free_1.gpkg",
    "transport":  "Philadelphia_gis_osm_transport_free_1.gpkg",

    # 线
    "roads":      "Philadelphia_gis_osm_roads_free_1.gpkg",
    "railways":   "Philadelphia_gis_osm_railways_free_1.gpkg",
    "waterways":  "Philadelphia_gis_osm_waterways_free_1.gpkg",

    # 面
    "buildings":  "Philadelphia_gis_osm_buildings_a_free_1.gpkg",
    "landuse":    "Philadelphia_gis_osm_landuse_a_free_1.gpkg",
    "natural":    "Philadelphia_gis_osm_natural_a_free_1.gpkg",
    "water":      "Philadelphia_gis_osm_water_a_free_1.gpkg",
}

OUT_CSV = "geo_feature_matrix.csv"

# ============================
# 2）读入商家点
# ============================
df = pd.read_csv(BUSINESS_FILE)

gdf_biz = gpd.GeoDataFrame(
    df[[ID_COL, LAT_COL, LON_COL]].copy(),
    geometry=gpd.points_from_xy(df[LON_COL], df[LAT_COL]),
    crs="EPSG:4326"
)

# 转成米制 CRS（UTM）
utm_crs = gdf_biz.estimate_utm_crs()
gdf_biz = gdf_biz.to_crs(utm_crs)

# ============================
# 3）生成 buffers
# ============================
buffers = {}
for r in RADII:
    b = gdf_biz[[ID_COL, "geometry"]].copy()
    b["geometry"] = b.geometry.buffer(r)
    buffers[r] = b

# ============================
# 4）核心统计函数
# ============================
def summarize_layer(layer_gdf, layer_name, kind):
    """
    kind: 'point' | 'line' | 'polygon'
    """
    results = []

    for r in RADII:
        buf = buffers[r]

        joined = gpd.sjoin(
            layer_gdf[["geometry"]],
            buf,
            predicate="intersects",
            how="inner"
        )

        # count
        out = joined.groupby(ID_COL).size().reset_index(
            name=f"{layer_name}_{r}m_count"
        )

        # 线：长度
        if kind == "line":
            joined["_len"] = joined.geometry.length
            length = joined.groupby(ID_COL)["_len"].sum().reset_index(
                name=f"{layer_name}_{r}m_len"
            )
            out = out.merge(length, on=ID_COL, how="left")

        # 面：面积
        if kind == "polygon":
            joined["_area"] = joined.geometry.area
            area = joined.groupby(ID_COL)["_area"].sum().reset_index(
                name=f"{layer_name}_{r}m_area"
            )
            out = out.merge(area, on=ID_COL, how="left")

        results.append(out)
        del joined
        gc.collect()

    # 合并不同半径
    merged = results[0]
    for r in results[1:]:
        merged = merged.merge(r, on=ID_COL, how="outer")

    return merged

# ============================
# 5）跑所有 OSM 图层
# ============================
features = gdf_biz[[ID_COL]].drop_duplicates().copy()

for layer_name, gpkg in OSM_GPKGS.items():
    print(f"Processing {layer_name} ...")

    if not os.path.exists(gpkg):
        print(f"  ⚠️ file not found, skip")
        continue

    layer = gpd.read_file(gpkg)
    layer = layer.to_crs(utm_crs)
    layer = layer[layer.geometry.notna() & ~layer.geometry.is_empty]

    if layer_name in ["roads", "railways", "waterways"]:
        kind = "line"
    elif layer_name in ["buildings", "landuse", "natural", "water"]:
        kind = "polygon"
    else:
        kind = "point"

    feat = summarize_layer(layer, layer_name, kind)
    features = features.merge(feat, on=ID_COL, how="left")

    del layer
    gc.collect()

# 缺失值 = 0
features = features.fillna(0)

# ============================
# 6）导出
# ============================
features.to_csv(OUT_CSV, index=False)
print("✅ Saved:", OUT_CSV)
print(features.head())