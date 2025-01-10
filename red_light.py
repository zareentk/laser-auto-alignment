import cv2
import numpy as np
#238, 50, 24 for ana 633 flourescent red 
#ff0000 in rgb
#Filter shade primary red
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 50, 50])
upper_red2 = np.array([180, 255, 255])


lower_green1 = np.array([40, 50, 50])
upper_green1 = np.array([75, 255, 255])

# Start video capture
cap = cv2.VideoCapture(0)

# Set exposure settings (adjust as needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -2)

MAX_BRIGHTNESS_RADIUS = 50

# Define the stationary LED position (adjust these coordinates as needed)
#stationary_x, stationary_y = 240, 240  # Example coordinates (center of frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red color
    r_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    r_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(r_mask1, r_mask2)

    green_mask = cv2.inRange(hsv, lower_green1, upper_green1)

    # Process the red mask with erosion and dilation to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.erode(red_mask, kernel, iterations=1)
    red_mask = cv2.dilate(red_mask, kernel, iterations=3)
    cap.set(cv2.CAP_PROP_GAIN, 5)

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
    # Find contours in the combined mask
    r_contours, _ = cv2.findContours(
        r_combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    g_contours, _ = cv2.findContours(
        g_combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # If a moving point is detected that is red, draw it and calculate displacement
    if r_contours and g_contours:
        cr = max(r_contours, key=cv2.contourArea)
        cg= max(g_contours, key=cv2.contourArea)
        ((xr, yr), red_radius) = cv2.minEnclosingCircle(cr)
        ((xg, yg), green_radius) = cv2.minEnclosingCircle(cg)

        # Draw a moving point for the red light
        cv2.circle(frame, (int(xr), int(yr)), int(red_radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(xr), int(yr)), 5, (0, 255, 0), -1)

        # Draw a moving point for the green light
        cv2.circle(frame, (int(xg), int(yg)), int(green_radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(xg), int(yg)), 5, (0, 0, 255), -1)

        # Calculate the displacement to the stationary LED
        displacement = np.sqrt((xr - xg) ** 2 + (yr - yg) ** 2)

        cv2.putText(
            frame,
            f"Red Light Position: ({int(xr)}, {int(yr)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Green Light Position: ({int(xg)}, {int(yg)})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Displacement between red and green LED: {displacement:.2f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw a line between the stationary and moving points
        cv2.line(frame, (int(xg), int(yg)), (int(xr), int(yr)), (0, 255, 0), 2)

    # Display the different masks and the result
    cv2.imshow("Red Light Tracking", frame)
    cv2.imshow("Brightness Mask Filtered", filtered_bright_mask)
    cv2.imshow("Red Combined Mask", r_combined_mask)
    cv2.imshow("Green Combined Mask", g_combined_mask)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Green Mask", green_mask)
    cv2.imshow("Brightness Mask", bright_mask)


    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()