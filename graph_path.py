import matplotlib.pyplot as plt
from multiprocessing import Process, Queue
import red_light  # Import the detection file

def plot_lasers(input_queue):
    # Initialize plot
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Laser Paths")
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.invert_yaxis()

    red_path_x = []
    red_path_y = []
    green_path_x = []
    green_path_y = []

    red_line, = ax.plot([], [], 'r-', label="Red Laser")
    green_line, = ax.plot([], [], 'g-', label="Green Laser")
    ax.legend()

    while True:
        # Retrieve data from the queue
        if not input_queue.empty():
            red_coords, green_coords = input_queue.get()

            if red_coords:
                red_path_x.append(red_coords[0])
                red_path_y.append(red_coords[1])
                red_line.set_data(red_path_x, red_path_y)

            if green_coords:
                green_path_x.append(green_coords[0])
                green_path_y.append(green_coords[1])
                green_line.set_data(green_path_x, green_path_y)

            # Update plot limits
            ax.set_xlim(0, 640)
            ax.set_ylim(0, 480)

            plt.pause(0.001)

if __name__ == "__main__":
    queue = Queue()
    detection_process = Process(target=red_light.detect_lasers, args=(queue,))
    plotting_process = Process(target=plot_lasers, args=(queue,))

    detection_process.start()
    plotting_process.start()

    detection_process.join()
    plotting_process.join()
