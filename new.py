from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper Functions
def process_image(image_bytes):
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return image, hsv_image

def green_mask(hsv_image):
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    return cv2.inRange(hsv_image, lower_green, upper_green)

def clean_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)

def calculate_average_rgb(segmented_image, mask):
    masked_region = segmented_image[mask != 0]
    return np.mean(masked_region, axis=0) if masked_region.size > 0 else [0, 0, 0]

def calculate_gi(average_rgb):
    return 2 * average_rgb[1] - (average_rgb[0] + average_rgb[2])

def calculate_ndvi(x):
    return 0.003 * x + 0.1087

def calculate_iey(ndvi, das):
    return ndvi / das if das > 0 else 0

def calculate_pyp(iey):
    return 6084.6 * (iey / 1.61)

def calculate_green_area_ratio(total_pixels, green_pixels):
    return green_pixels / total_pixels if total_pixels > 0 else 0

def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string

@app.post("/process-images/")
async def process_images(
    healthy_image: UploadFile = File(...),
    limited_image: UploadFile = File(...),
    das: float = Form(...),
):
    try:
        print("Sania's backend is running")
        # Read and process healthy image
        healthy_image_bytes = await healthy_image.read()
        healthy_rgb, healthy_hsv = process_image(healthy_image_bytes)
        mask_healthy = green_mask(healthy_hsv)
        cleaned_mask_healthy = clean_mask(mask_healthy)
        segmented_healthy = healthy_rgb.copy()
        segmented_healthy[cleaned_mask_healthy == 0] = 0

        # Calculate total pixels (A) and green pixels (B) for healthy image
        total_pixels_healthy = int(healthy_rgb.shape[0] * healthy_rgb.shape[1])  # Total number of pixels (A)
        green_pixels_healthy = int(np.sum(cleaned_mask_healthy > 0))  # Count non-zero pixels (B)
        green_area_ratio_healthy = calculate_green_area_ratio(total_pixels_healthy, green_pixels_healthy)  # B/A

        # Calculate GI, X, and NDVI for healthy image
        avg_rgb_healthy = calculate_average_rgb(segmented_healthy, cleaned_mask_healthy)
        gi_healthy = calculate_gi(avg_rgb_healthy)
        x_healthy = gi_healthy * green_area_ratio_healthy  # X = GI * (B/A)
        ndvi_healthy = calculate_ndvi(x_healthy)

        # Read and process nitrogen-limited image
        limited_image_bytes = await limited_image.read()
        limited_rgb, limited_hsv = process_image(limited_image_bytes)
        mask_limited = green_mask(limited_hsv)
        cleaned_mask_limited = clean_mask(mask_limited)
        segmented_limited = limited_rgb.copy()
        segmented_limited[cleaned_mask_limited == 0] = 0

        # Calculate total pixels (A) and green pixels (B) for limited image
        total_pixels_limited = int(limited_rgb.shape[0] * limited_rgb.shape[1])  # Total number of pixels (A)
        green_pixels_limited = int(np.sum(cleaned_mask_limited > 0))  # Count non-zero pixels (B)
        green_area_ratio_limited = calculate_green_area_ratio(total_pixels_limited, green_pixels_limited)  # B/A

        # Calculate GI, X, and NDVI for limited image
        avg_rgb_limited = calculate_average_rgb(segmented_limited, cleaned_mask_limited)
        gi_limited = calculate_gi(avg_rgb_limited)
        x_limited = gi_limited * green_area_ratio_limited  # X = GI * (B/A)
        ndvi_limited = calculate_ndvi(x_limited)
        iey_limited = calculate_iey(ndvi_limited, das)
        pyp_limited = calculate_pyp(iey_limited)

        # Convert nitrogen-limited image to base64
        nitrogen_limited_base64 = image_to_base64(segmented_limited)
        segmented_healthy_base64 = image_to_base64(segmented_healthy)

        # Additional calculations
        ri = ndvi_healthy / ndvi_limited if ndvi_limited != 0 else 0  # Relative Index
        pypn = pyp_limited * ri  # Potential Yield Production for Nitrogen
        nitrogen_rate = 10 * (pypn - pyp_limited) * (1.85 / 50)  # Nitrogen Rate in kg/acre

        # Fertilizer calculations
        urea_needed = nitrogen_rate * 2.1739  # Urea Needed
        can_needed = nitrogen_rate * 3.8461  # CAN Needed
        ammonium_sulfate_needed = nitrogen_rate * 4.7619  # Ammonium Sulfate Needed

        # Return results as JSON
        return JSONResponse(
            {
                "healthy_plot": {
                    "Total Pixels (A)": total_pixels_healthy,
                    "Green Pixels (B)": green_pixels_healthy,
                    "NDVI": round(float(ndvi_healthy), 4),
                    "GI": round(float(gi_healthy), 2),
                    "X (GI * B/A)": round(float(x_healthy), 2),
                },
                "nitrogen_limited_plot": {
                    "Total Pixels (A)": total_pixels_limited,
                    "Green Pixels (B)": green_pixels_limited,
                    "NDVI": round(float(ndvi_limited), 4),
                    "GI": round(float(gi_limited), 2),
                    "X (GI * B/A)": round(float(x_limited), 2),
                    "IEY (kg/ha)": round(float(iey_limited), 2),
                    "PYP (kg/ha)": round(float(pyp_limited), 2),
                },
                "additional_calculations": {
                    "RI (Relative Index)": round(float(ri), 4),
                    "PYPN (Potential Yield for Nitrogen)": round(float(pypn), 2),
                    "Nitrogen Rate (kg/acre)": round(float(nitrogen_rate), 2),
                },
                "fertilizer_recommendations": {
                    "Urea Needed (kg/acre)": round(float(urea_needed), 2),
                    "CAN Needed (kg/acre)": round(float(can_needed), 2),
                    "Ammonium Sulfate Needed (kg/acre)": round(float(ammonium_sulfate_needed), 2),
                },
                "nitrogen_limited_base64": nitrogen_limited_base64,
                "segmented_healthy_base64":segmented_healthy_base64,
            }
        )
    except Exception as e:
        return JSONResponse({"error": str(e)})