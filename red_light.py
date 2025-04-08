import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
from collections import deque
from utils import Control_Algorithm

from ultralytics import YOLO
import csv
import torch
lower_red1 = np.array([0, 15, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 15, 50])
upper_red2 = np.array([180, 255, 255])

#initialize servo positions
microseconds_servo_x = 1833
microseconds_servo_y = 1060
prev_microseconds_x = 1833
prev_microseconds_y = 1060

# Start video capture (change to 0 or 1 depending on camera used)
cap = cv2.VideoCapture(0)
#cap.set(cv2.CAP_PROP_FOCUS, 8)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Set exposure settings (adjust s needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

MAX_BRIGHTNESS_RADIUS = 50

car_positions = []


#move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# Queue for graphing positions
red_positions = deque(maxlen=100)
model = YOLO("Car_Laser5.pt")
model.to(device)

if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

while True: 
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red LED
    r_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    r_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(r_mask1, r_mask2)

    # Process the red mask with erosion and dilation to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.erode(red_mask, kernel, iterations=1)
    red_mask = cv2.dilate(red_mask, kernel, iterations=3)

    # Create a brightness mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Filter the brightness mask to keep only bright spots of certain size
    filtered_bright_mask = np.zeros_like(bright_mask)
    contours, _ = cv2.findContours(
        bright_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:
        ((xr, yr), red_radius) = cv2.minEnclosingCircle(c)
        ##Limits radius of laser spot (red)
        if red_radius <= MAX_BRIGHTNESS_RADIUS:
            cv2.drawContours(filtered_bright_mask, [c], -1, 255, thickness=cv2.FILLED)

        
    # Combine the red and brightness masks
    r_combined_mask = cv2.bitwise_and(red_mask, filtered_bright_mask)

    red_coords = None

    # Find contours in the combined mask
    r_contours, _ = cv2.findContours(
        r_combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    results = model(frame)
    car_center = None
    prev_car_center = 1575 , 544 
    car_box = None

    results = model(frame)
    car_center = None
    prev_car_center = 1575, 544 
    car_box = None
    red_coords = None
    CONF_THRESH = 0.15
 
    if results and results[0].boxes is not None:
        names = results[0].names
        boxes = results[0].boxes
        car_boxes = []
        laser_boxes = []
 
        def box_area(b):
            x1, y1, x2, y2 = b
            return (x2 - x1) * (y2 - y1)
 
        for box in boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue  # Skip low confidence detections
 
            cls = int(box.cls[0])
            label = names.get(cls, '')
            coords = box.xyxy[0].cpu().numpy()
            
            if label == 'car':
                car_boxes.append(coords)
            elif label == 'Red-laser':
                laser_boxes.append(coords)
 
    if car_boxes:
        car_box = max(car_boxes, key=box_area)
        x1, y1, x2, y2 = car_box.astype(int)
        car_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, car_center, 5, (0, 0, 255), -1)
 
    if laser_boxes:
        red_box = max(laser_boxes, key=box_area)
        xr1, yr1, xr2, yr2 = red_box.astype(int)
        red_coords = (int((xr1 + xr2) / 2), int((yr1 + yr2) / 2))
        cv2.rectangle(frame, (xr1, yr1), (xr2, yr2), (255, 0, 0), 2)
        cv2.circle(frame, red_coords, 5, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"Red Light Position: {red_coords}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        red_positions.append(red_coords)

    #time.sleep(0.25)
    #If car is detected then we find the displacement if its not detected we go to default
    if car_center is not None:
        if red_coords is not None:
            displacement = np.sqrt((red_coords[0] - car_center[0]) ** 2 + (red_coords[1] - car_center[1]) ** 2)
            microseconds_servo_x , microseconds_servo_y = Control_Algorithm(red_coords[0], red_coords[1], car_center[0], car_center[1], prev_car_center[0], prev_car_center[1],prev_microseconds_x, prev_microseconds_y)
            #send_value(1000, 544)
            prev_microseconds_x = microseconds_servo_x
            prev_microseconds_y = microseconds_servo_y
            prev_car_center = car_center
            
            
            cv2.putText(
                frame,
                f"Displacement between red LED and car: {displacement:.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.line(frame, (int(car_center[0]), int(car_center[1])), (int(red_coords[0]), int(red_coords[1])), (0, 255, 0), 2)
            

        else:
            pass

    else:
        pass

    # Display the frame
    cv2.imshow("LED Tracking", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()


with open('Car_positions_csv.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'X', 'Y'])  # CSV Header
    writer.writerows(car_positions)  # Write all stored data
