import os
import subprocess
import numpy as np

# Paths
input_path = "input/"
output_path = "output/"
executable = "./recon_ring_remover_v3"

# Parameters for the C++ program
first_image_num = 1
last_image_num = 16
center_x, center_y = 0, 0
ring_width = 30
thresh_min, thresh_max = -300, 300
threshold = 300
angular_min = 30
verbose = 1

def run_cpp_program(image_num):
    command = [
        executable,
        input_path,
        output_path,
        "rec_",
        "out_openmp_",
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

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running C++ program for image {image_num}: {result.stderr}")
        return None, None

    output_lines = result.stdout.split('\n')
    serial_time = parallel_time = None
    for line in output_lines:
        if "Serial filtering time:" in line:
            serial_time = float(line.split(":")[1].strip().split()[0])
        elif "Parallel filtering time:" in line:
            parallel_time = float(line.split(":")[1].strip().split()[0])

    return serial_time, parallel_time

serial_times = {i: [] for i in range(first_image_num, last_image_num + 1)}
parallel_times = {i: [] for i in range(first_image_num, last_image_num + 1)}

print(f"{'Image':<10}{'Serial Time (s)':<20}{'Parallel Time (s)':<20}{'Speedup':<10}")
print("-" * 60)

with open('openmp_results.txt', 'w') as f:
    f.write(f"{'Image':<10}{'Serial Time (s)':<20}{'Parallel Time (s)':<20}{'Speedup':<10}\n")
    f.write("-" * 60 + "\n")
    
    for i in range(first_image_num, last_image_num + 1):
        for run_num in range(1, 21):
            serial_time, parallel_time = run_cpp_program(i)
            if serial_time is not None and parallel_time is not None:
                serial_times[i].append(serial_time)
                parallel_times[i].append(parallel_time)
        
        avg_serial_time = np.mean(serial_times[i])
        avg_parallel_time = np.mean(parallel_times[i])
        speedup = avg_serial_time / avg_parallel_time if avg_parallel_time > 0 else float('inf')
        
        print(f"{i:<10}{avg_serial_time:<20.4f}{avg_parallel_time:<20.4f}{speedup:<10.2f}")
        f.write(f"{i:<10}{avg_serial_time:<20.4f}{avg_parallel_time:<20.4f}{speedup:<10.2f}\n")

print("\nOpenMP results saved to openmp_results.txt")