from ultralytics import YOLO
import cv2
import time

print("Loading model...")
# Load model
model = YOLO("models/model3/runs/train/weights/best.pt")

print("Model loaded.")

# Open webcam
cap = cv2.VideoCapture(0)

while True:

    # Read frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to get camera frame")
        break

    # YOLO detection
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show image
    cv2.imshow("YOLO Detection", annotated_frame)

        # Check if any objects were detected

    result = results[0]

    if len(result.boxes) == 0:
        print("No objects detected.")

    else:
        # Loop through every detected object
        for i, box in enumerate(result.boxes):

            # ------------------------------
            # Class ID
            # ------------------------------
            class_id = int(box.cls[0])

            # Convert class ID into class name
            class_name = model.names[class_id]

            # ------------------------------
            # Confidence
            # ------------------------------
            confidence = float(box.conf[0])

            # ------------------------------
            # Bounding Box Coordinates
            # xyxy = left, top, right, bottom
            # ------------------------------
            x1, y1, x2, y2 = box.xyxy[0]

            # Convert tensors into normal numbers
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            # ------------------------------
            # Calculate centre point
            # ------------------------------
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ------------------------------
            # Width and Height
            # ------------------------------
            width = x2 - x1
            height = y2 - y1

            # ------------------------------
            # Print everything
            # ------------------------------
            print(f"Object #{i+1}")
            print(f"Class ID      : {class_id}")
            print(f"Class Name    : {class_name}")
            print(f"Confidence    : {confidence:.2f}")
            print(f"Bounding Box  : ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"Width         : {width}")
            print(f"Height        : {height}")
            print(f"Centre        : ({center_x}, {center_y})")
            print()

    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()