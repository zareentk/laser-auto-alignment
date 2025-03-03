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

# Start video capture
cap = cv2.VideoCapture(1)

# Set exposure settings (adjust as needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)

MAX_BRIGHTNESS_RADIUS = 50

# Kalman filters for red LEDs
kf_red = KalmanFilter()

# Queue for graphing positions
red_positions = deque(maxlen=100)
model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(1)

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

    frame = cv2.flip(frame, 1)

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

    # Detect and track LED positions
    if r_contours:
        cr = max(r_contours, key=cv2.contourArea)
        ((xr, yr), red_radius) = cv2.minEnclosingCircle(cr)
        kf_red.correct(xr, yr)
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
    else:
        pred_red = kf_red.predict()
        red_coords = (pred_red[0][0], pred_red[1][0])
        red_positions.append(red_coords)

    # Run YOLO detection
    results = model(frame)

    #For object detection of the car
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            class_name = model.names[cls]

            # Try detecting it as a regular "car"
            if class_name == "car" and conf > 0.3:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Car ({conf:.2f})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                car_center = ((x1+x2)/2, (y1+y2)/2)
                cv2.putText(
                frame,
                f"Car Position: ({(car_center)})",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6,
                (255, 255, 255),
                2,
                )
            else:
                car_center = (0,0)

    time.sleep(0.25)
    #If car is detected then we find the displacement if its not detected we go to default
    if (car_center != (0,0)):
        displacement = np.sqrt((red_coords[0] - car_center[0]) ** 2 + (red_coords[1] - car_center[1]) ** 2)
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
        # Initialize the PIDController with gain values of 1 for both axes
        pid = PIDController(Kp_x=0, Ki_x=0, Kd_x=0.1, Kp_y=0.1, Ki_y=0.1, Kd_y=0.1, limit_out=100)

        # Compute the PID corrections
        output_x, output_y = pid.correct(red_coords[0], red_coords[1], car_center[0], car_center[1]) #output is a correction factor that needs to be transformed into a servo motor position, corresponding to a pwm

        # Print the outputs
        print(f"PID output for X-axis: {output_x}")
        print(f"PID output for Y-axis: {output_y}")
        send_value(output_x)
    else:
        ##make it so it goes to center
        send_value(0,0)

    # Update the live plot
    if red_positions:
        red_x, red_y = zip(*red_positions)
        red_line.set_data(red_x, red_y)

    ax.set_xlim(0, frame.shape[1])
    ax.set_ylim(0, frame.shape[0])
    plt.pause(0.01)

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
plt.ioff()
plt.show()
