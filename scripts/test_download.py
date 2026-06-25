import ee
import geemap

ee.Initialize(project="isro-ps10")

image = (
    ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
    .filterDate('2024-01-01', '2024-12-31')
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .first()
)

print(image.get('LANDSAT_PRODUCT_ID').getInfo())

geemap.ee_export_image(
    image.select(['SR_B2']),
    filename=r"C:\Users\Vinay Singh\Desktop\test_B2.tif",
    scale=30,
    file_per_band=False
)

print("DONE")