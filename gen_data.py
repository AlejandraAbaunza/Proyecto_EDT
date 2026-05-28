import rasterio
import geopandas as gpd
import pandas as pd
import numpy as np
from rasterio.mask import mask

# -----------------------------
# ARCHIVOS DE ENTRADA
# -----------------------------
raster_path = "bandas_stack_clip.tif"
shapefile_path = "training_proy2_geometries.shp"

# -----------------------------
# CARGAR DATOS
# -----------------------------
gdf = gpd.read_file(shapefile_path)
src = rasterio.open(raster_path)

rows = []

# -----------------------------
# LOOP POR POLÍGONOS (ROIs)
# -----------------------------
for _, row in gdf.iterrows():

    geom = [row.geometry]

    # OJO: nombre de columna de clase (puede variar)
    label = row.iloc[-1]

    out_image, out_transform = mask(src, geom, crop=True)

    bands, height, width = out_image.shape

    for i in range(height):
        for j in range(width):

            pixel = out_image[:, i, j]

            # ignorar nodata o ceros
            if np.any(pixel == 0):
                continue

            lon, lat = rasterio.transform.xy(out_transform, i, j)

            rows.append([
                lat, lon,
                pixel[0], pixel[1], pixel[2],
                pixel[3], pixel[4], pixel[5],
                pixel[6], pixel[7],
                label
            ])

# -----------------------------
# CREAR DATAFRAME FINAL
# -----------------------------
df = pd.DataFrame(rows, columns=[
    "Latitude", "Longitude",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11",
    "Class"
])

# -----------------------------
# EXPORTAR TSV
# -----------------------------
df.to_csv("dataset.tsv", sep="\t", index=False)

print("✔ Dataset TSV generado correctamente")