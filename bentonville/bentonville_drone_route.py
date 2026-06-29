

import math, heapq, os, sys
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import osmnx as ox
import pyproj
from rasterio.features import rasterize
from rasterio.transform import from_bounds


grid = 600
cache = "bentonville_osm.gpkg"
start_coordinates = (36.3236, -94.2556)   # Walmart drone industrial area
end_coordinates    = (36.3854, -94.2256)   # Bentonville residential (can change location to go through nature areas))

noise_costs = {
    "industrial": 8, "commercial": 20, "wood": 30, "wetland": 35,
    "residential": 50, "farmland": 55, "religious": 75,
    "school": 100, "hospital": 200,
}
fill_cost = 1

zone_colors = {
    "industrial": "#b5b5b5", "commercial": "#f4a460", "wood": "#3a7d44",
    "wetland": "#5b8db8", "residential": "#f08080", "farmland": "#c5e1a5",
    "religious": "#ba68c8", "school": "#ef5350", "hospital": "#b71c1c",
    "other": "#e8e8e8",
}

osm_tags = {
    "landuse": ["residential", "industrial", "commercial", "religious", "farmland", "forest"],
    "amenity": ["school", "hospital", "place_of_worship"],
    "natural":  ["wetland", "wood"],
}


# OSM load 

if os.path.exists(cache):
    gdf = gpd.read_file(cache)
else:
    gdf = ox.features_from_place("Bentonville, Arkansas, USA", osm_tags)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    for col in gdf.columns:
        if gdf[col].apply(lambda v: isinstance(v, list)).any():
            gdf = gdf.drop(columns=[col])
    gdf.to_file(cache, driver="GPKG")

gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()


# zones

def classify(row):
    a = row.get("amenity")
    if a == "school":           return "school"
    if a == "hospital":         return "hospital"
    if a == "place_of_worship": return "religious"
    lu = row.get("landuse")
    lu_map = {"residential": "residential", "industrial": "industrial",
              "commercial": "commercial", "religious": "religious",
              "farmland": "farmland", "forest": "wood"}
    if lu in lu_map: return lu_map[lu]
    n = row.get("natural")
    if n == "wetland": return "wetland"
    if n == "wood":    return "wood"
    return "other"

gdf["zone"] = gdf.apply(classify, axis=1)
gdf["cost"] = gdf["zone"].map(noise_costs)
gdf = gdf.dropna(subset=["cost"]).copy()

print("\nZone breakdown:")
print(gdf["zone"].value_counts().to_string())

utm = gdf.estimate_utm_crs()
gdf = gdf.to_crs(utm)


# zone map

fig, ax = plt.subplots(figsize=(12, 12))
for zone, grp in gdf.groupby("zone"):
    grp.plot(ax=ax, color=zone_colors.get(zone, "#e8e8e8"), edgecolor="none", alpha=0.85, label=zone)
ax.legend(title="Zone", loc="lower right", fontsize=9, framealpha=0.8)
ax.set_title("Bentonville, AR — Noise Sensitivity Zones", fontsize=14)
ax.set_axis_off()
plt.savefig("bentonville_zones.png", dpi=300, bbox_inches="tight")
plt.close()


# cost raster

xmin, ymin, xmax, ymax = gdf.total_bounds
cw = (xmax - xmin) / grid   # cell width  (m)
ch = (ymax - ymin) / grid   # cell height (m)
tfm = from_bounds(xmin, ymin, xmax, ymax, grid, grid)

valid = set(noise_costs.values())
gdf_r = gdf[gdf["cost"].isin(valid)].sort_values("cost", ascending=True).copy()

cost_grid = rasterize(
    list(zip(gdf_r.geometry, gdf_r["cost"].astype("float32"))),
    out_shape=(grid, grid), transform=tfm, fill=float(fill_cost), dtype="float32",
)

print(f"\nRaster shape: {cost_grid.shape}")
print(f"Unique costs: {np.unique(cost_grid)}")

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(cost_grid, origin="upper", cmap="magma_r")
plt.colorbar(im, ax=ax, label="Noise Sensitivity Cost")
ax.set_title("Bentonville — Noise Cost Surface", fontsize=13)
ax.axis("off")
plt.savefig("bentonville_cost_surface.png", dpi=300, bbox_inches="tight")
plt.close()


# coordinate helpers

to_utm   = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
to_wgs84 = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)

def ll_to_px(lat, lon):
    ux, uy = to_utm.transform(lon, lat)
    r = int(np.clip((ymax - uy) / (ymax - ymin) * grid, 0, grid - 1))
    c = int(np.clip((ux - xmin) / (xmax - xmin) * grid, 0, grid - 1))
    return r, c

def px_to_ll(r, c):
    ux = xmin + (c + 0.5) / grid * (xmax - xmin)
    uy = ymax - (r + 0.5) / grid * (ymax - ymin)
    lon, lat = to_wgs84.transform(ux, uy)
    return lat, lon

start_px = ll_to_px(*start_coordinates)
goal_px  = ll_to_px(*end_coordinates)

# A* pathfinding

def astar(grid, start, goal):
    rows, cols = grid.shape
    moves = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    min_c = 1.0
    heap = [(0.0, start)]
    came_from, g = {}, {start: 0.0}
    while heap:
        _, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur); cur = came_from[cur]
            return [start] + path[::-1]
        for dr, dc in moves:
            nr, nc = cur[0]+dr, cur[1]+dc
            if not (0 <= nr < rows and 0 <= nc < cols): continue
            nb = (nr, nc)
            tg = g[cur] + grid[nr, nc] * math.sqrt((dr*ch)**2 + (dc*cw)**2)
            if nb not in g or tg < g[nb]:
                came_from[nb] = cur; g[nb] = tg
                h = math.sqrt(((nr-goal[0])*ch)**2 + ((nc-goal[1])*cw)**2) * min_c
                heapq.heappush(heap, (tg + h, nb))
    return None


noise_path = astar(cost_grid, start_px, goal_px)


short_path = astar(np.ones_like(cost_grid), start_px, goal_px)

if noise_path is None or short_path is None:
    print("ERROR: no path found — check that both endpoints fall inside the raster.")
    sys.exit(1)


# metrics

def measure(path, grid):
    dist = noise = 0.0
    for i in range(1, len(path)):
        dr, dc = path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]
        s = math.sqrt((dr*ch)**2 + (dc*cw)**2)
        dist += s; noise += grid[path[i][0], path[i][1]] * s
    return dist, noise

n_dist, n_noise = measure(noise_path, cost_grid)
s_dist, s_noise = measure(short_path, cost_grid)

noise_reduction = (s_noise - n_noise) / s_noise * 100
extra_dist      = (n_dist  - s_dist)  / s_dist  * 100

print(f"\nNoise-aware:  {n_dist/1000:.2f} km  |  noise exposure = {n_noise:.0f}")
print(f"Shortest:     {s_dist/1000:.2f} km  |  noise exposure = {s_noise:.0f}")
print(f"\nNoise reduction:  {noise_reduction:.1f}%")
print(f"Distance penalty: {extra_dist:.1f}% longer")


# single noise-aware A* path plot

norm_grid = (cost_grid - cost_grid.min()) / (cost_grid.max() - cost_grid.min())

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(norm_grid, origin="upper", cmap="viridis")
plt.colorbar(ax.images[0], ax=ax, label="Noise Cost")
ax.plot([p[1] for p in noise_path], [p[0] for p in noise_path],
        color="blue", linewidth=2, label="Route")
ax.scatter(start_px[1], start_px[0], color="green", s=120, zorder=6, label="Start (Walmart DC)")
ax.scatter(goal_px[1],  goal_px[0],  color="red",  s=120, marker="x", linewidths=2.5, zorder=6,
           label="End (Residential)")
ax.set_title("Bentonville – Noise-Minimised Drone Route", fontsize=13)
ax.legend(loc="upper right", fontsize=9, framealpha=0.85)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("bentonville_astar_path.png", dpi=150, bbox_inches="tight")
plt.close()


# route comparison plott

fig, ax = plt.subplots(figsize=(13, 13))
ax.imshow(cost_grid, origin="upper", cmap="magma_r", alpha=0.72)

ax.plot([p[1] for p in short_path], [p[0] for p in short_path],
        color="white", linewidth=2, linestyle="--", alpha=0.85,
        label=f"Shortest  ({s_dist/1000:.2f} km, cost {s_noise:.0f})")
ax.plot([p[1] for p in noise_path], [p[0] for p in noise_path],
        color="deepskyblue", linewidth=2.5,
        label=f"Noise-aware  ({n_dist/1000:.2f} km, cost {n_noise:.0f})")

ax.scatter(start_px[1], start_px[0], color="lime", s=160, zorder=6, label="Start — Walmart Drone DC")
ax.scatter(goal_px[1],  goal_px[0],  color="red",  s=160, marker="x", linewidths=2.5, zorder=6,
           label="End — NE Bentonville residential")

ax.set_title(f"Bentonville UAV Route Comparison\n"
             f"Noise reduction: {noise_reduction:.1f}%  |  Extra distance: {extra_dist:.1f}%", fontsize=13)
ax.legend(loc="upper left", fontsize=10, framealpha=0.85)
ax.annotate("N\n↑", xy=(0.96, 0.96), xycoords="axes fraction", fontsize=13,
            fontweight="bold", ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
ax.set_axis_off()
plt.tight_layout()
plt.savefig("bentonville_route_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: bentonville_zones.png, bentonville_cost_surface.png, bentonville_route_comparison.png")


# waypoints (for ardupilot)

waypoints = [px_to_ll(r, c) for r, c in noise_path]
print(f"\nNoise-aware path — first 5 waypoints (lat, lon):")
for lat, lon in waypoints[:5]:
    print(f"  {lat:.6f}, {lon:.6f}")
print(f"  ... ({len(waypoints)} total)")