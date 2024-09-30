import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Paths
input_path = r"../../../images/input"
output_path = r"images/output"
executable = "./ring_remover.exe"  # Assuming executable is in current directory

# Parameters for the C++ program
first_image_num = 1
last_image_num = 16  # Adjust when more images are available
center_x = 0
center_y = 0
ring_width = 30
thresh_min = -300
thresh_max = 300
threshold = 300
angular_min = 30
verbose = 1

# Function to run the C++ program and extract CPU and GPU times
def run_cpp_program(image_num):
    command = [
        executable,
        input_path,
        output_path,
        "rec_",
        "test_out",  # Changed output prefix to match the new format
        str(image_num),
        str(image_num),
        str(center_x),
        str(center_y),
        str(ring_width),
        str(thresh_min),
        str(thresh_max),
        str(threshold),
        str(angular_min),
        str(verbose)
    ]

    # Run the C++ program and capture its output
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running C++ program for image {image_num}: {result.stderr}")
        return None, None

    # Parse the output to extract CPU and GPU times
    output_lines = result.stdout.split('\n')
    cpu_time = None
    gpu_time = None
    for line in output_lines:
        if "CPU ring filtering time:" in line:
            cpu_time = float(line.split(":")[1].strip().split()[0])
        elif "GPU ring filtering time:" in line:
            gpu_time = float(line.split(":")[1].strip().split()[0])

    return cpu_time, gpu_time

# Record times for CPU and GPU
cpu_times = {i: [] for i in range(first_image_num, last_image_num + 1)}
gpu_times = {i: [] for i in range(first_image_num, last_image_num + 1)}

# Run for each image multiple times
print(f"------------------------------------------------------------------------------------")
for i in range(first_image_num, last_image_num + 1):
    print(f"\nRing Removal Computation Time for Image {i} [rec_{i}.tif]:")
    print(f"{'Run':<5} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<10}")

    for run_num in range(1, 21):  # Number of run times, adjust as needed
        cpu_time, gpu_time = run_cpp_program(i)
        if cpu_time is not None and gpu_time is not None:
            cpu_times[i].append(cpu_time)
            gpu_times[i].append(gpu_time)
            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"{run_num:<5} {cpu_time:<15.4f} {gpu_time:<15.4f} {speedup:<.2f}x")

    # Calculate and print average times after all runs for the current image
    avg_cpu_time = np.mean(cpu_times[i])
    avg_gpu_time = np.mean(gpu_times[i])
    avg_speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
    print(f"\nAverage for Image {i} [rec_{i}.tif]:")
    print(f"  Average CPU time: {avg_cpu_time:.4f} seconds")
    print(f"  Average GPU time: {avg_gpu_time:.4f} seconds")
    print(f"  Average Speedup: {avg_speedup:.2f}x\n")
    print(f"------------------------------------------------------------------------------------")

# Create the Tkinter window with scrollable content
root = tk.Tk()
root.title("CPU vs GPU (OpenCL) Computation Time")
root.geometry("1800x800")  # Set window size to 1800x800 pixels

# Define a function to handle window close event
def on_closing():
    root.quit()  # This will stop the Tkinter mainloop and allow the program to exit

# Bind the window close event (clicking the 'X' button) to the on_closing function
root.protocol("WM_DELETE_WINDOW", on_closing)

# Create a frame for the canvas and the scrollbar
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

# Create a canvas
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# Add a scrollbar to the canvas
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Configure the canvas
canvas.configure(yscrollcommand=scrollbar.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

# Create another frame inside the canvas
second_frame = tk.Frame(canvas)

# Add that new frame to a window in the canvas
canvas.create_window((0, 0), window=second_frame, anchor="nw")

# Number of columns in each row
n_cols = 4
n_rows = (last_image_num + n_cols - 1) // n_cols  # Calculate number of rows based on the number of columns

# Create the figure and axes for the plots with a size of 1800x800 pixels
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(18, 8))

# Flatten the axs array to make it easier to index
axs = axs.flatten()

for i in range(first_image_num, last_image_num + 1):
    runs = range(1, len(cpu_times[i]) + 1)

    # Select the current subplot (each image has its own graph)
    ax = axs[i - 1] if last_image_num > 1 else axs  # Handle single subplot case

    ax.plot(runs, cpu_times[i], label=f'CPU Time (Image {i})', color='blue')
    ax.plot(runs, gpu_times[i], label=f'GPU Time (Image {i})', color='orange', linestyle='--')

    ax.set_xlabel('Run Number')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(f'CPU vs GPU Computation Time for Image {i}')
    ax.legend()

# Adjust layout and spacing
plt.subplots_adjust(hspace=1.0, wspace=1.0)

# Render the plot on a canvas
canvas_plot = FigureCanvasTkAgg(fig, second_frame)
canvas_plot.draw()
canvas_plot.get_tk_widget().pack()

# Start the Tkinter event loop
root.mainloop()
