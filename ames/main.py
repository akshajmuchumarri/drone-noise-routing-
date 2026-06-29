import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
import heapq

from rasterio.features import rasterize
from rasterio.transform import from_bounds

# Download data from OpenStreetMap



tags = {
    "landuse": [
        "residential",
        "industrial",
        "commercial"
    ],
    "amenity": [
        "school",
        "hospital"
    ]
}


gdf = ox.features_from_place(
    "Ames, Iowa, USA",
    tags
)



# Create a unified zone column
gdf["zone"] = "other"

# Land use categories
gdf.loc[gdf["landuse"] == "residential", "zone"] = "residential"
gdf.loc[gdf["landuse"] == "industrial", "zone"] = "industrial"
gdf.loc[gdf["landuse"] == "commercial", "zone"] = "commercial"

# Amenities override land-use
gdf.loc[gdf["amenity"] == "school", "zone"] = "school"
gdf.loc[gdf["amenity"] == "hospital", "zone"] = "hospital"

# Keep only polygons (ignore points for now)
gdf = gdf[
    gdf.geometry.geom_type.isin(
        ["Polygon", "MultiPolygon"]
    )
]

# Show category counts
#print("\nZone counts:")
#print(gdf["zone"].value_counts())

# Plot
fig, ax = plt.subplots(figsize=(12, 12))

gdf.plot(
    column="zone",
    categorical=True,
    legend=True,
    ax=ax
)

ax.set_title("Ames, Iowa - Noise Sensitivity Zones")
ax.set_axis_off()

plt.savefig(
    "ames_zones.png",
    dpi=300,
    bbox_inches="tight"
)

cost = {
    "industrial": 5,
    "commercial": 20,
    "residential": 50,
    "school": 100,
    "hospital": 200
}

gdf["cost"] = gdf["zone"].map(cost)

# Remove anything without a valid cost
gdf = gdf.dropna(subset=["cost"])


# Convert coordinates to a projected CRS (meters)


gdf = gdf.to_crs(gdf.estimate_utm_crs())


# Create rasterization shapes


shapes = [
    (geom, cost)
    for geom, cost in zip(gdf.geometry, gdf.cost)
]


# Define raster dimensions


xmin, ymin, xmax, ymax = gdf.total_bounds

width = 500
height = 500

transform = from_bounds(
    xmin,
    ymin,
    xmax,
    ymax,
    width,
    height
)


# Rasterize polygons into a NumPy array


raster = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=1,
    dtype="float32"
)


# Basic diagnostics


print("Raster shape:", raster.shape)
print("Unique costs:", np.unique(raster))


# Save raster visualization


plt.figure(figsize=(10, 10))

plt.imshow(
    raster,
    origin="lower"
)

plt.colorbar(label="Cost")

plt.title("Ames Noise Sensitivity Map")

plt.savefig(
    "cost_surface.png",
    dpi=300,
    bbox_inches="tight"
)


# testing example

start = (30,30)
goal = (350, 400)

#distance

def heuristic(a, b):
    return math.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2
    )

#actual A*

def astar(raster, start, goal):

    rows, cols = raster.shape

    directions = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
        (-1, -1), (-1, 1),
        (1, -1), (1, 1)
    ]

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {start: 0}

    while open_set:

        current = heapq.heappop(open_set)[1]

        if current == goal:

            path = []

            while current in came_from:
                path.append(current)
                current = came_from[current]

            path.append(start)

            return path[::-1]

        for dr, dc in directions:

            nr = current[0] + dr
            nc = current[1] + dc

            if not (
                0 <= nr < rows and
                0 <= nc < cols
            ):
                continue

            neighbor = (nr, nc)

            step_distance = math.sqrt(
                dr*dr + dc*dc
            )

            traversal_cost = raster[nr, nc]

            tentative_g = (
                g_score[current]
                + traversal_cost * step_distance
            )

            if (
                neighbor not in g_score
                or
                tentative_g < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                f_score = (
                    tentative_g
                    + heuristic(
                        neighbor,
                        goal
                    )
                )

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return None


#running the algorithm

path = astar(
    raster,
    start,
    goal
)

print ("Path len:", len(path))

#plot

plt.figure(figsize=(10, 10))

plt.imshow(
    raster,
    origin="lower"
)

rows = [p[0] for p in path]
cols = [p[1] for p in path]

plt.plot(
    cols,
    rows,
    linewidth=2
)

plt.scatter(
    start[1],
    start[0],
    marker="o"
)

plt.scatter(
    goal[1],
    goal[0],
    marker="x"
)

plt.title("Noise-Minimized Path")

plt.colorbar()

plt.savefig(
    "astar_path.png",
    dpi=300,
    bbox_inches="tight"
)

#convert to lat + long