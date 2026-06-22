import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import cv2
import easyocr
from ultralytics import YOLO


MODEL_PATH = "models/gunny_bag_detector.pt"
OUTPUT_DIR = "outputs/ocr_results"

model = YOLO(MODEL_PATH)
reader = easyocr.Reader(["en"], gpu=False)


def detect_and_read(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Could not read image:", image_path)
        return

    results = model(image)[0]

    bag_count = 0
    extracted_texts = []

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

            if bag_crop.size == 0:
                continue

            ocr_result = reader.readtext(bag_crop)
            text = " ".join([item[1] for item in ocr_result])

            if text.strip():
                extracted_texts.append(text)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"gunny_bag {conf:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    if bag_count == 0:
        status = "No Gunny Bag Found"
    elif len(extracted_texts) == 0:
        status = "ALERT: No Label/Text Found"
    else:
        status = "Label/Text Found"

    cv2.putText(
        image,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255) if "ALERT" in status else (0, 255, 0),
        2,
    )

    print("Image:", image_path)
    print("Total Gunny Bags:", bag_count)
    print("Status:", status)
    print("Extracted Text:", extracted_texts)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    image_name = os.path.basename(image_path)
    output_path = os.path.join(OUTPUT_DIR, "result_" + image_name)

    cv2.imwrite(output_path, image)
    print("Output saved to:", output_path)

    return {
        "image": image_path,
        "bag_count": bag_count,
        "status": status,
        "extracted_text": extracted_texts,
        "output_path": output_path,
    }


if __name__ == "__main__":
    folder = "dataset/yolo_dataset/test/images"

    image_files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("Total images found:", len(image_files))

    # Testing only first 10 images because OCR is slow on CPU
    for image_name in image_files[:10]:
        print("\n==============================")
        print("Processing:", image_name)

        image_path = os.path.join(folder, image_name)
        detect_and_read(image_path)

    print("\nDone. Results saved in:", OUTPUT_DIR)
  
