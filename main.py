"""
License Plate Detection & Recognition Pipeline
===============================================
Fixes applied:
    1. Frame resized for display (window too large)
    2. Q key now reliably quits (waitKey fix)
    3. Annotations drawn even when OCR returns None
    4. Verbose YOLO logs suppressed (verbose=False)
    5. Empty detection guard for SORT tracker

Dependencies:
    pip install ultralytics opencv-python numpy
    # SORT: https://github.com/abewley/sort
"""

from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort
from util import get_car, read_license_plate, write_csv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VEHICLE_CLASSES = [2, 3, 5, 7]   # COCO: car, motorcycle, bus, truck
DISPLAY_WIDTH   = 1280            # Display window width  (change to taste)
DISPLAY_HEIGHT  = 720             # Display window height (change to taste)


# ---------------------------------------------------------------------------
# Load models
# verbose=False suppresses the per-frame speed logs you were seeing
# ---------------------------------------------------------------------------
coco_model             = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

mot_tracker = Sort()
cap         = cv2.VideoCapture('./sample.mp4')

orig_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

results   = {}
frame_nmr = -1

# ---------------------------------------------------------------------------
# FIX 1: Create a NAMED, NORMAL window before the loop
# This is required for cv2.waitKey to reliably catch keypresses on most
# systems. Without WINDOW_NORMAL, Q often gets missed.
# ---------------------------------------------------------------------------
cv2.namedWindow("License Plate Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("License Plate Detection", DISPLAY_WIDTH, DISPLAY_HEIGHT)

print(f"Video: {orig_width}x{orig_height} | Display window: {DISPLAY_WIDTH}x{DISPLAY_HEIGHT}")
print("Press Q (or Escape) in the video window to quit.\n")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_nmr += 1
    results[frame_nmr] = {}

    # -----------------------------------------------------------------------
    # Step 1: Detect vehicles
    # verbose=False stops the "0: 384x640 18 cars..." logs from printing
    # -----------------------------------------------------------------------
    detections  = coco_model(frame, verbose=False)[0]
    detections_ = []

    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in VEHICLE_CLASSES:
            detections_.append([x1, y1, x2, y2, score])

    # -----------------------------------------------------------------------
    # Step 2: Update SORT tracker
    # Guard against empty detections — SORT crashes on an empty list
    # -----------------------------------------------------------------------
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = mot_tracker.update(np.empty((0, 5)))

    # -----------------------------------------------------------------------
    # Step 3: Detect license plates
    # -----------------------------------------------------------------------
    license_plates = license_plate_detector(frame, verbose=False)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # -------------------------------------------------------------------
        # Step 4: Match plate to a tracked vehicle
        # -------------------------------------------------------------------
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        if car_id == -1:
            continue

        # -------------------------------------------------------------------
        # Step 5: Crop plate region — guard against bad bounding boxes
        # -------------------------------------------------------------------
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]

        if license_plate_crop.size == 0:
            continue

        # -------------------------------------------------------------------
        # Step 6: Preprocess for OCR
        # -------------------------------------------------------------------
        gray   = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        gray   = cv2.equalizeHist(gray)
        gray   = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # -------------------------------------------------------------------
        # Step 7: OCR
        # -------------------------------------------------------------------
        license_plate_text, license_plate_text_score = read_license_plate(thresh)

        # -------------------------------------------------------------------
        # FIX 2: Draw vehicle box regardless of OCR result
        # Previously boxes were only drawn inside "if text is not None",
        # so nothing appeared when OCR failed
        # -------------------------------------------------------------------

        # Green box + ID around the vehicle
        cv2.rectangle(frame,
            (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)),
            (0, 255, 0), 3)
        cv2.putText(frame,
            f"Car {int(car_id)}",
            (int(xcar1), max(int(ycar1) - 12, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Red box around the license plate
        cv2.rectangle(frame,
            (int(x1), int(y1)), (int(x2), int(y2)),
            (0, 0, 255), 3)

        if license_plate_text is not None:
            # Red plate text above the plate box
            cv2.putText(frame,
                license_plate_text,
                (int(x1), max(int(y1) - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Print to terminal
            print(
                f"Frame {frame_nmr:>4} | "
                f"Car {int(car_id):>3} | "
                f"Plate: {license_plate_text:<12} | "
                f"Conf: {license_plate_text_score:.2f}"
            )

            # Save to results dict
            results[frame_nmr][car_id] = {
                'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                'license_plate': {
                    'bbox':       [x1, y1, x2, y2],
                    'text':        license_plate_text,
                    'bbox_score':  score,
                    'text_score':  license_plate_text_score
                }
            }
        else:
            # OCR failed but plate was detected — show placeholder
            cv2.putText(frame,
                "Plate?",
                (int(x1), max(int(y1) - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2)

    # -----------------------------------------------------------------------
    # Display full-res frame — the named window scales it to DISPLAY size
    # -----------------------------------------------------------------------
    cv2.imshow("License Plate Detection", frame)

    # -----------------------------------------------------------------------
    # FIX 3: Use waitKey(25) instead of waitKey(1)
    # waitKey(1) gives the OS only 1ms to register a keypress — too short
    # on many systems. 25ms (~40fps) is reliable without slowing video much.
    # Also handle Escape (27) as an alternative quit key.
    # -----------------------------------------------------------------------
    key = cv2.waitKey(2) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        print("Quit signal received.")
        break

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
cap.release()
cv2.destroyAllWindows()

write_csv(results, './test.csv')
total = sum(len(v) for v in results.values())
print(f"\nDone. {total} plate detections saved to test.csv")