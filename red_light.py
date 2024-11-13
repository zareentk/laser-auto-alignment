import cv2
import numpy as np
#238, 50, 24 for ana 633 flourescent red 
#ff0000 in rgb
#Filter shade primary red
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([15, 255, 255])
lower_red2 = np.array([150, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Start video capture
cap = cv2.VideoCapture(0)

# Set exposure settings (adjust as needed for lighting)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

MAX_BRIGHTNESS_RADIUS = 20

# Define the stationary LED position (adjust these coordinates as needed)
stationary_x, stationary_y = 240, 240  # Example coordinates (center of frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for detecting red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

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
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        if radius <= MAX_BRIGHTNESS_RADIUS:
            cv2.drawContours(filtered_bright_mask, [c], -1, 255, thickness=cv2.FILLED)

    # Combine the red and brightness masks
    combined_mask = cv2.bitwise_and(red_mask, filtered_bright_mask)

    # Find contours in the combined mask
    contours, _ = cv2.findContours(
        combined_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Draw the stationary LED point
    cv2.circle(frame, (stationary_x, stationary_y), 5, (255, 0, 0), -1)
    cv2.putText(
        frame,
        f"Stationary LED Position: ({stationary_x}, {stationary_y})",
        (stationary_x + 10, stationary_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 0),
        1,
    )

    # If a moving point is detected, draw it and calculate displacement
    if contours:
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Draw the moving point
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Calculate the displacement to the stationary LED
        displacement = np.sqrt((x - stationary_x) ** 2 + (y - stationary_y) ** 2)

        cv2.putText(
            frame,
            f"Red Light Position: ({int(x)}, {int(y)})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Displacement to Stationary LED: {displacement:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw a line between the stationary and moving points
        cv2.line(frame, (stationary_x, stationary_y), (int(x), int(y)), (0, 255, 0), 2)

    # Display the different masks and the result
    cv2.imshow("Red Light Tracking", frame)
    cv2.imshow("Brightness Mask Filtered", filtered_bright_mask)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Red Mask", red_mask)
    cv2.imshow("Brightness Mask", bright_mask)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()