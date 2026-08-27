from ultralytics import YOLO
import cv2
import serial
import time
import tkinter as tk
import yaml

#Set to fals if not using arduino
using_arduino = True

#Arduino With Yolo

if using_arduino == True :
    print("Connecting to Arduino...")
    arduino = serial.Serial('COM11', 9600)
    last_sent = 99
    time.sleep(2)
    print("Arduino connected.")

print("Lodaing tkinter...")

root = tk.Tk()
root.title("YOLO Detection Data")
root.geometry("400x500")
root.resizable(True, True)

highest_object = tk.Label(
    root,
    text="No Objects Detected",
    font=("Arial", 14),
    justify="left"
)
highest_object.pack()

object_confidence=tk.Label(
    root,
    text = "No Object Detected",
    font = ("Arial", 14),
    justify="left"
)
object_confidence.pack()

camera_state = tk.Label(
    root,
    text="Camera: Awaiting Connection",
    font=("Arial", 14),
    justify="left"
)
camera_state.pack()

arduino_signal = tk.Label(
    root,
    text="No Signal Sent", 
    font=("Arial", 14),
    justify="left"
)
arduino_signal.pack()

print("Tkinter loaded.")

print("Loading yaml...")

with open("models/model3/data.yaml", "r") as file:
    data = yaml.safe_load(file)

classes = data["names"]

print(classes)

print("Yaml loaded.")

print("Loading color visualzation...")

object_colors = {
    "Glass": "#00FF00",
    "Metal": "#FF0000",
    "Paper": "#0000FF",
    "Plastic": "#FFFF00"
}

print("Color visualzation loaded.")

print("Loading model...")
# Load model
model = YOLO("models/model3/runs/train/weights/best.pt")

print("Model loaded.")


# Open webcam
print("Opening webcam...")
camera = cv2.VideoCapture(0)
print("Webcam opened.")



while True:

    # Read camera
    connection_success, frame = camera.read()

    #Check connection success, if not then alert user
    if connection_success:
        camera_state.config(text="Camera: Connected")
    else:
        print("Failed to get camera frame")
        camera_state.config(text="Camera: Not Connected")
    
    #Send camera data to YOLO model
    results = model(frame)

    # Draw boxes and labels
    annotated_frame = results[0].plot()

    # Show data on OpenCV window
    cv2.imshow("YOLO Detection", annotated_frame)

    result = results[0]

    

    if len(result.boxes) == 0:
        item_detected = False
        highest_object.config(text="No object detected",
                              fg="black")
        object_confidence.config(text="No object detected")
        print("No objects detected.")

        if using_arduino == True:
            signal = 10
            if signal != last_sent:
                last_sent = signal
                arduino.write(bytes([signal]))
                arduino.flush()
                arduino_signal.config(text=signal)

    else:
        # Loop through every detected object
        item_detected = True
        detected_item = {}
        highest_confidence = 0

        for i, box in enumerate(result.boxes):            

            # Class ID
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1
            print(f"Object #{i+1}")
            print(f"Class ID      : {class_id}")
            print(f"Class Name    : {class_name}")
            print(f"Confidence    : {confidence:.2f}")
            print(f"Bounding Box  : ({x1}, {y1}) -> ({x2}, {y2})")
            print(f"Width         : {width}")
            print(f"Height        : {height}")
            print(f"Centre        : ({center_x}, {center_y})")
            print()

            if confidence > highest_confidence:
                highest_confidence = confidence
                
                detected_item ={
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height
                }

        highest_object.config(
            text=detected_item["class_name"],
            fg=object_colors[detected_item["class_name"]])
        object_confidence.config(
            text=detected_item["confidence"])


        signal = detected_item["class_id"]
        if using_arduino == True:
            if highest_confidence > 0.5: #signal and signal != last_sent and 
                if signal != last_sent:
                    arduino.write(bytes([signal]))
                    arduino.flush()
                    last_sent = signal
                    arduino_signal.config(text=signal)
                #last_action = now#if now - last_action > cooldown:
                #arduino.write(f"{signal+1}\n".encode('utf-8'))
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    

    
    root.update()

        