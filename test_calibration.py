import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils import KalmanFilter
from utils import send_value
from utils import PIDController
from ultralytics import YOLO
import time
#238, 50, 24 for ana 633 flourescent red 
#ff0000 in rgb  
#Filter shade primary red
##GO TO DEFAULT IMMEDIATELY IF NOTHING IS DETECTED
lower_red1 = np.array([0, 15, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 15, 50])
upper_red2 = np.array([180, 255, 255])

# Start video capture (change to 0 or 1 depending on camera used)
cap = cv2.VideoCapture(1)


# Set resolution to **1280x720** (720p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Set exposure settings (adjust as needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -6)

MAX_BRIGHTNESS_RADIUS = 50

# Kalman filters for red LEDs
#kf_red = KalmanFilter()

# List to store data before writing
car_positions = []

# Queue for graphing positions
red_positions = deque(maxlen=100)
model = YOLO("yellow.pt")

if not cap.isOpened():
    print("Error: Unable to access the camera")
    exit()

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

# Path plotting
red_line, = ax.plot([], [], 'r-', label='Red LED Path')
ax.legend()
ax.set_title('LED Position Tracking')
ax.set_xlim(0, 640)  
ax.set_ylim(0, 480) 
ax.invert_yaxis()
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')

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
        bright_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    #Record coordinates and print them out 
        # Detect and track LED positions
    if r_contours:
        cr = max(r_contours, key=cv2.contourArea)
        ((xr, yr), red_radius) = cv2.minEnclosingCircle(cr)
        red_coords = (xr, yr)
        red_positions.append(red_coords)
        cv2.circle(frame, (int(xr), int(yr)), int(red_radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(xr), int(yr)), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            f"Red Light Position: ({int(xr)}, {int(yr)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        time.sleep(1)
        print(red_coords)
        send_value(red_coords[0], red_coords[1])

    # Display the frame
    cv2.imshow("LED Tracking", frame)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Bright Mask", bright_mask)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# Release resources
cap.release()
cv2.destroyAllWindows()



