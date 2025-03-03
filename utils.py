import cv2
import numpy as np
import serial
import time
import csv
import os
# Kalman filter setup
# Set up Serial communication with Arduino
arduino =serial.Serial(port='COM8', baudrate=115200, timeout=.1)   # Replace 'COM3' with your Arduino's port
class KalmanFilter:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

    def predict(self):
        return self.kf.predict()

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        return self.kf.correct(measurement)
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
    
def compute_angular_error(x_laser, x_receiver, image_width=1280, fov_x=70):
   
    # Convert pixel positions to angular positions
    theta_laser = ((2 * x_laser / image_width) - 1) * (fov_x / 2)
    theta_receiver = ((2 * x_receiver / image_width) - 1) * (fov_x / 2)

    # Compute angular error
    angular_error = theta_laser - theta_receiver

    return angular_error



def send_value(value):
    arduino.write(bytes(str(value) + '\n', 'utf-8'))  # Append '\n' so Arduino reads properly
    data = arduino.readline().decode().strip()  # Read response from Arduino
    print("Received from Arduino:", data)