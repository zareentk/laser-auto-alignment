import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils import KalmanFilter
from utils import send_value
from utils import PIDController
#238, 50, 24 for ana 633 flourescent red 
#ff0000 in rgb
#Filter shade primary red

lower_red1 = np.array([0, 15, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 15, 50])
upper_red2 = np.array([180, 255, 255])


lower_green1 = np.array([40, 50, 50])
upper_green1 = np.array([75, 255, 255])


# Start video capture
cap = cv2.VideoCapture(1)

# Set exposure settings (adjust as needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -7)

MAX_BRIGHTNESS_RADIUS = 50

# Kalman filters for red and green LEDs
kf_red = KalmanFilter()
kf_green = KalmanFilter()

# Queue for graphing positions
red_positions = deque(maxlen=100)
green_positions = deque(maxlen=100)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

# Path plotting
red_line, = ax.plot([], [], 'r-', label='Red LED Path')
green_line, = ax.plot([], [], 'g-', label='Green LED Path')
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

    # Create masks for detecting red and green colors
    r_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    r_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(r_mask1, r_mask2)

    green_mask = cv2.inRange(hsv, lower_green1, upper_green1)

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
    g_combined_mask = cv2.bitwise_and(green_mask, filtered_bright_mask)

    red_coords = None
    green_coords = None

    # Find contours in the combined mask
    r_contours, _ = cv2.findContours(
        bright_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    g_contours, _ = cv2.findContours(
        g_combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

    if g_contours:
        cg = max(g_contours, key=cv2.contourArea)
        ((xg, yg), green_radius) = cv2.minEnclosingCircle(cg)
        kf_green.correct(xg, yg)
        green_coords = (xg, yg)
        green_positions.append(green_coords)
        cv2.circle(frame, (int(xg), int(yg)), int(green_radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(xg), int(yg)), 5, (0, 0, 255), -1)
        cv2.putText(
            frame,
            f"Green Light Position: ({int(xg)}, {int(yg)})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
    else:
        pred_green = kf_green.predict()
        green_coords = (pred_green[0][0], pred_green[1][0])
        green_positions.append(green_coords)

    if red_coords and green_coords:
        displacement = np.sqrt((red_coords[0] - green_coords[0]) ** 2 + (red_coords[1] - green_coords[1]) ** 2)
        cv2.putText(
            frame,
            f"Displacement between red and green LED: {displacement:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.line(frame, (int(green_coords[0]), int(green_coords[1])), (int(red_coords[0]), int(red_coords[1])), (0, 255, 0), 2)
        # Initialize the PIDController with gain values of 1 for both axes
        pid = PIDController(Kp_x=0, Ki_x=0, Kd_x=0.1, Kp_y=0.1, Ki_y=0.1, Kd_y=0.1, limit_out=100)

        # Compute the PID corrections
        output_x, output_y = pid.correct(red_coords[0], red_coords[1], green_coords[0], green_coords[1]) #output is a correction factor that needs to be transformed into a servo motor position, corresponding to a pwm

        # Print the outputs
        print(f"PID output for X-axis: {output_x}")
        print(f"PID output for Y-axis: {output_y}")
        send_value(output_x)
    # Update the live plot
    if red_positions:
        red_x, red_y = zip(*red_positions)
        red_line.set_data(red_x, red_y)

    if green_positions:
        green_x, green_y = zip(*green_positions)
        green_line.set_data(green_x, green_y)

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
