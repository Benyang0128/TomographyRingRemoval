import os
import subprocess
import numpy as np

# Paths
input_path = r"C:\Users\PC\source\repos\CudaRuntime4\CudaRuntime4\InputImages"
output_path = r"C:\Users\PC\source\repos\CudaRuntime4\CudaRuntime4\OutputImages"
executable = "./ring_remover.exe"

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
        "test_out",
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
    cpu_time = gpu_time = None
    for line in output_lines:
        if "CPU ring filtering time:" in line:
            cpu_time = float(line.split(":")[1].strip().split()[0])
        elif "GPU ring filtering time:" in line:
            gpu_time = float(line.split(":")[1].strip().split()[0])
    
    return cpu_time, gpu_time

cpu_times = {i: [] for i in range(first_image_num, last_image_num + 1)}
gpu_times = {i: [] for i in range(first_image_num, last_image_num + 1)}

print(f"{'Image':<10}{'CPU Time (s)':<15}{'GPU Time (s)':<15}{'Speedup':<10}")
print("-" * 50)

with open('cuda_results.txt', 'w') as f:
    f.write(f"{'Image':<10}{'CPU Time (s)':<15}{'GPU Time (s)':<15}{'Speedup':<10}\n")
    f.write("-" * 50 + "\n")
    
    for i in range(first_image_num, last_image_num + 1):
        for run_num in range(1, 21):
            cpu_time, gpu_time = run_cpp_program(i)
            if cpu_time is not None and gpu_time is not None:
                cpu_times[i].append(cpu_time)
                gpu_times[i].append(gpu_time)
        
        avg_cpu_time = np.mean(cpu_times[i])
        avg_gpu_time = np.mean(gpu_times[i])
        speedup = avg_cpu_time / avg_gpu_time if avg_gpu_time > 0 else float('inf')
        
        print(f"{i:<10}{avg_cpu_time:<15.4f}{avg_gpu_time:<15.4f}{speedup:<10.2f}")
        f.write(f"{i:<10}{avg_cpu_time:<15.4f}{avg_gpu_time:<15.4f}{speedup:<10.2f}\n")

print("\nCUDA results saved to cuda_results.txt")