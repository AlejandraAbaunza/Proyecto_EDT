import rasterio
import numpy as np
import joblib
import pandas as pd


# --------------------------------------------------
# 1. Cargar modelo y scaler
# --------------------------------------------------

model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------------
# 2. Abrir stack Sentinel-2
# --------------------------------------------------

stack_file = "bandas_stack_clip.tif"

with rasterio.open(stack_file) as src:

    stack = src.read()

    profile = src.profile

    height = src.height
    width = src.width

print("Dimensiones:", height, width)
print("Bandas:", stack.shape[0])

# --------------------------------------------------
# 3. Convertir raster a matriz de píxeles
# --------------------------------------------------

n_bands = stack.shape[0]

pixels = stack.reshape(
    n_bands,
    height * width
).T

print("Pixels:", pixels.shape)

# --------------------------------------------------
# 4. Detectar NoData
# --------------------------------------------------

valid_mask = np.all(
    pixels > 0,
    axis=1
)

valid_pixels = pixels[valid_mask]

print("Pixeles válidos:", len(valid_pixels))

# --------------------------------------------------
# 5. Escalar
# --------------------------------------------------

valid_pixels_df = pd.DataFrame(
    valid_pixels,
    columns=[
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B11"
    ]
)

valid_pixels_scaled = scaler.transform(
    valid_pixels_df
)

# --------------------------------------------------
# 6. Clasificar
# --------------------------------------------------

predictions = model.predict(
    valid_pixels_scaled
)

# --------------------------------------------------
# 7. Reconstruir raster
# --------------------------------------------------

classified = np.zeros(
    height * width,
    dtype=np.uint8
)

classified[valid_mask] = predictions

classified = classified.reshape(
    height,
    width
)

# --------------------------------------------------
# 8. Guardar GeoTIFF
# --------------------------------------------------

profile.update(
    count=1,
    dtype=rasterio.uint8,
    nodata=0
)

output_file = "clasificacion_svm.tif"

with rasterio.open(
    output_file,
    "w",
    **profile
) as dst:

    dst.write(
        classified,
        1
    )

print("\nMapa clasificado guardado:")
print(output_file)