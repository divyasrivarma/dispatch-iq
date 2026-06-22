import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import easyocr
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO


MODEL_PATH = "models/gunny_bag_detector.pt"
OUTPUT_DIR = "outputs/api_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(
    title="Dispatch IQ Backend",
    description="Gunny bag detection and OCR API",
    version="1.0"
)

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)


@app.get("/")
def home():
    return {
        "message": "Dispatch IQ Backend is running",
        "endpoints": {
            "image_detection": "/detect-image"
        }
    }


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, file.filename)

        with open(input_path, "wb") as f:
            f.write(await file.read())

        image = cv2.imread(input_path)

        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": "Could not read uploaded image"
                }
            )

        results = model(image)[0]

        bag_count = 0
        extracted_texts = []
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

            if class_name == "gunny_bag" and conf > 0.3:
                bag_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = image.shape[:2]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                bag_crop = image[y1:y2, x1:x2]

                text = ""

                if bag_crop.size > 0:
                    ocr_result = reader.readtext(bag_crop)
                    text = " ".join([item[1] for item in ocr_result])

                    if text.strip():
                        extracted_texts.append(text)

                detections.append({
                    "class": class_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "text": text
                })

                cv2.rectangle(image, (x1, y1), (x2, y2), (22, 163, 74), 2)
                cv2.putText(
                    image,
                    f"gunny_bag {conf:.2f}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (22, 163, 74),
                    2,
                )

        if bag_count == 0:
            final_status = "No Gunny Bag Found"
        elif len(extracted_texts) == 0:
            final_status = "ALERT: No Label/Text Found"
        else:
            final_status = "Label/Text Found"

        output_path = os.path.join(OUTPUT_DIR, "api_result.jpg")
        cv2.imwrite(output_path, image)

        return {
            "status": "success",
            "input_file": file.filename,
            "bag_count": bag_count,
            "label_status": final_status,
            "extracted_text": extracted_texts,
            "detections": detections,
            "output_path": output_path
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
