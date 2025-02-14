import cv2
import numpy as np
import serial
import time
# Kalman filter setup
# Set up Serial communication with Arduino
arduino = serial.Serial(port='COM8', baudrate=115200, timeout=.1)   # Replace 'COM3' with your Arduino's port
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

    def correct(self, target_x, current_x, target_y, current_y, dt=None):
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

        return output_x, output_y

# Function to send data to Arduino
def send_value(value):
        arduino.write(bytes(str(value), 'utf-8')) 
        time.sleep(0.1)
        data = arduino.readlines()
        print(data)