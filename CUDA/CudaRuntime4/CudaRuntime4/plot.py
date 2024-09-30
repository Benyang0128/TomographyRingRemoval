import matplotlib.pyplot as plt
import numpy as np

def read_results(filename):
    images = []
    times1 = []
    times2 = []
    with open(filename, 'r') as f:
        next(f)  # Skip header
        next(f)  # Skip separator line
        for line in f:
            parts = line.strip().split()
            image = int(parts[0])
            time1 = float(parts[1])
            time2 = float(parts[2])
            images.append(image)
            times1.append(time1)
            times2.append(time2)
    return images, times1, times2

cuda_images, _, cuda_gpu_times = read_results('cuda_results.txt')
opencl_images, _, opencl_gpu_times = read_results('opencl_results.txt')
openmp_images, openmp_cpu_times, openmp_parallel_times = read_results('openmp_results.txt')

plt.figure(figsize=(12, 8))

# Plot CPU time (using OpenMP's CPU time)
plt.plot(openmp_images, openmp_cpu_times, 'k-', linewidth=1)
plt.scatter(openmp_images, openmp_cpu_times, color='k', marker='o', s=50, label='CPU (Serial)')

# Plot GPU/Parallel times for each method
plt.plot(cuda_images, cuda_gpu_times, 'b-', linewidth=1)
plt.scatter(cuda_images, cuda_gpu_times, color='b', marker='o', s=50, label='CUDA GPU')

plt.plot(opencl_images, opencl_gpu_times, 'r-', linewidth=1)
plt.scatter(opencl_images, opencl_gpu_times, color='r', marker='o', s=50, label='OpenCL GPU')

plt.plot(openmp_images, openmp_parallel_times, 'g-', linewidth=1)
plt.scatter(openmp_images, openmp_parallel_times, color='g', marker='o', s=50, label='OpenMP Parallel')

plt.xlabel('Image Number')
plt.ylabel('Time (seconds)')
plt.title('Comparison of Computation Times')
plt.legend()
plt.grid(True)

# Set x-axis ticks to be integers
plt.xticks(range(min(openmp_images), max(openmp_images)+1))

# Adjust y-axis limits to bring lines closer
all_times = openmp_cpu_times + cuda_gpu_times + opencl_gpu_times + openmp_parallel_times
y_min = max(0, min(all_times) * 0.9)  # Set minimum to 0 or 90% of the lowest time
y_max = max(all_times) * 1.1  # Set maximum to 110% of the highest time
plt.ylim(y_min, y_max)

plt.savefig('comparison_plot_openmp_cpu.png', dpi=300, bbox_inches='tight')
plt.show()

print("Comparison plot saved as comparison_plot_openmp_cpu.png")