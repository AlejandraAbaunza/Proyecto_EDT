import pandas as pd
import numpy as np
import rasterio

with rasterio.open("clasificacion_svm.tif") as src:

    raster = src.read(1)

    pixel_size_x = src.transform[0]
    pixel_size_y = abs(src.transform[4])

pixel_area = pixel_size_x * pixel_size_y  # m²

classes = np.unique(raster)

results = []

total_pixels = 0

for c in classes:

    if c == 0:   # NoData
        continue

    count = np.sum(raster == c)

    total_pixels += count

    area_m2 = count * pixel_area
    area_ha = area_m2 / 10000
    area_km2 = area_m2 / 1_000_000

    results.append([
        c,
        count,
        area_ha,
        area_km2
    ])

area_df = pd.DataFrame(
    results,
    columns=[
        "Clase",
        "Pixeles",
        "Area_ha",
        "Area_km2"
    ]
)

# -------------------------
# Área total clasificada
# -------------------------
total_area_ha = area_df["Area_ha"].sum()
total_area_km2 = area_df["Area_km2"].sum()

# -------------------------
# Porcentaje por clase
# -------------------------
area_df["Porcentaje"] = (
    area_df["Area_ha"] /
    total_area_ha * 100
)

print("\nÁreas por clase:")
print(area_df)

print("\nÁrea total clasificada:")
print(f"{total_area_ha:.2f} ha")
print(f"{total_area_km2:.2f} km²")

# -------------------------
# Guardar CSV
# -------------------------
area_df.to_csv(
    "areas_por_clase.csv",
    index=False
)

# -------------------------
# Resumen total
# -------------------------
resumen = pd.DataFrame({
    "Total_pixeles": [total_pixels],
    "Total_area_ha": [total_area_ha],
    "Total_area_km2": [total_area_km2]
})

#resumen.to_csv(
#    "area_total.csv",
#    index=False
#)