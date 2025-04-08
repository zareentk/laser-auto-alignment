import cv2
import numpy as np
import serial
import time
import csv
import os

# Set up Serial communication with Arduino
arduino =serial.Serial(port='COM10', baudrate=115200, timeout=.1)   # Replace 'COM10' with your Arduino's port
class PIDController:
    '''
    2-Axis PID Controller
    '''
    def __init__(self, Kp_x, Ki_x, Kd_x, Kp_y, Ki_y, Kd_y, limit_out=None):
        '''
        Initialize PID parameters for both axes.
        :param Kp_x, Ki_x, Kd_x: PID gains for the x-axis
        :param Kp_y, Ki_y, Kd_y: PID gains for the y-axis
        :param limit_out: Output limitation (range [-limit_out, limit_out])
        '''
        self.prev_time = 0
        self.prev_error_x = 0
        self.prev_error_y = 0

        # PID Gains for x and y axes
        self.Kp_x, self.Ki_x, self.Kd_x = Kp_x, Ki_x, Kd_x
        self.Kp_y, self.Ki_y, self.Kd_y = Kp_y, Ki_y, Kd_y

        # Cumulative errors for integration
        self.cumulative_error_x = 0.0
        self.cumulative_error_y = 0.0

        # Output limit
        self.limit_out = limit_out

    def correct(self, current_x, current_y, target_x, target_y, dt=None):
        '''
        Compute PID corrections for both axes.
        :param target_x, target_y: Desired values (setpoints) for x and y axes
        :param current_x, current_y: Current values (sensor readings) for x and y axes
        :param dt: Time step (use constant or compute dynamically)
        :return: PID correction values for x and y axes
        '''
        error_x = target_x - current_x
        error_y = target_y - current_y

        now = time.time()

        # Compute time delta
        if dt is None:
            dt = now - self.prev_time
        # Avoid division by zero
        if dt > 0:
            # Integration terms
            self.cumulative_error_x += error_x * dt
            self.cumulative_error_y += error_y * dt

            # Derivative terms
            derivative_error_x = (error_x - self.prev_error_x) / dt
            derivative_error_y = (error_y - self.prev_error_y) / dt
        else:
            derivative_error_x = 0
            derivative_error_y = 0

        # Compute PID outputs
        output_x = (self.Kp_x * error_x) + (self.Ki_x * self.cumulative_error_x) + (self.Kd_x * derivative_error_x)
        output_y = (self.Kp_y * error_y) + (self.Ki_y * self.cumulative_error_y) + (self.Kd_y * derivative_error_y)

        # Apply output limits if specified
        if self.limit_out is not None:
            output_x = max(min(output_x, self.limit_out), -self.limit_out)
            output_y = max(min(output_y, self.limit_out), -self.limit_out)

        # Update previous values
        self.prev_error_x = error_x
        self.prev_error_y = error_y
        self.prev_time = now

        # Define the Excel file name
        excel_file = "pid_log.csv"

        # Prepare the data row
        data_row = [
            now, target_x, current_x, error_x, self.cumulative_error_x, derivative_error_x,
            target_y, current_y, error_y, self.cumulative_error_y, derivative_error_y, dt,
            self.Kp_x, self.Ki_x, self.Kd_x, self.Kp_y, self.Ki_y, self.Kd_y,
            self.Kp_x * error_x, self.Ki_x * self.cumulative_error_x, self.Kd_x * derivative_error_x,
            self.Kp_y * error_y, self.Ki_y * self.cumulative_error_y, self.Kd_y * derivative_error_y,
            output_x, output_y
            ]
        # Check if the file exists to determine if we need a header
        file_exists = os.path.isfile(excel_file)

        # Write data to CSV file
        with open(excel_file, 'a', newline='') as file:
            writer = csv.writer(file)
            # Write header only if the file is new
            if not file_exists:
                writer.writerow([
                    "Time", "Target X", "Current X", "Error X", "Integral Error X", "Derivative Error X",
                    "Target Y", "Current Y", "Error Y", "Integral Error Y", "Derivative Error Y", "dt",
                    "Kp_x", "Ki_x", "Kd_x", "Kp_y", "Ki_y", "Kd_y",
                    "P_x", "I_x", "D_x", "P_y", "I_y", "D_y",
                    "Output X", "Output Y"
                    ])
            # Write the current data row
            writer.writerow(data_row)

        return output_x, output_y

camera_points = np.array([
    [92,4], [1097, 17], [1099, 650], [143, 656]], dtype=np.float32)

# Define corresponding servo angle positions (Destination)
servo_points = np.array([
    [2000,1150], [1450, 1150], [1450, 800], [2000, 800]], dtype=np.float32)

# Compute the perspective transformation matrix
M = cv2.getPerspectiveTransform(camera_points, servo_points)

# Function to map a single (x, y) point
def map_camera_to_servo(x, y):
    transformed = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)
    servo_x, servo_y = transformed[0][0]  # Extract single values
    return int(servo_x), int(servo_y)

def OnetoOne(prev_car_x, prev_car_y, car_x, car_y, laser_x, laser_y):
    # Convert all pixel positions to servo coordinates using homography
    prev_car_servo = map_camera_to_servo(prev_car_x, prev_car_y)
    curr_car_servo = map_camera_to_servo(car_x, car_y)
    laser_servo = map_camera_to_servo(laser_x, laser_y)
    
    # Compute how much the car has moved in servo space
    delta_servo_x = curr_car_servo[0] - prev_car_servo[0]
    delta_servo_y = curr_car_servo[1] - prev_car_servo[1]

    # Compute the new laser target position
    target_laser_servo_x = laser_servo[0] + delta_servo_x
    target_laser_servo_y = laser_servo[1] + delta_servo_y
    
    print(curr_car_servo[0])
    print(prev_car_servo[0])
    
    print(curr_car_servo[1])
    print(prev_car_servo[1])
    print(delta_servo_x)
    print(delta_servo_y)
    print(target_laser_servo_x)
    print(target_laser_servo_y)
    return (target_laser_servo_x,target_laser_servo_y)

# Initialize the PIDController with gain values of 1 for both axes
pid = PIDController(Kp_x=0.14, Ki_x=0, Kd_x=0.016, Kp_y=0.14, Ki_y=0, Kd_y=0.016, limit_out=100)

def Control_Algorithm(Laser_x, Laser_y, Car_x, Car_y, prev_Car, prev_microseconds_x, prev_microseconds_y):
    LED = 0
    distance = ((Car_x - Laser_x)**2 + (Car_y - Laser_y)**2)**0.5
    if distance > 200: #THRESHOLD_HOMOGRAPHY
        servo_x, servo_y = map_camera_to_servo(Car_x, Car_y) 
        print("homography")
    else: # distance > 400 or (distance <= 400 and prev_Car is None) or (distance <= 400 and car_changex == 0 and car_changey == 0): #THRESHOLD_PID
        PID_output_x, PID_output_y = pid.correct(Laser_x, Laser_y, Car_x, Car_y)
        servo_x = int(prev_microseconds_x - PID_output_x)
        servo_y = int(prev_microseconds_y - PID_output_y)
        print("PID") 
    if distance < 50:
        LED = 1
    servo_x = max(1445, min(servo_x, 2175))
    servo_y = max(745, min(servo_y, 1180))
    return servo_x,servo_y, LED, distance

def send_value(value1, value2, value3):
    arduino.write(bytes(str(value1) + "," + str(value2) + "," + str(value3) + '\n', 'utf-8'))  # Append '\n' so Arduino reads properly
    time.sleep(0.001)
    data = arduino.readline().decode().strip()  # Read response from Arduino
    print("Received from Arduino:", data)