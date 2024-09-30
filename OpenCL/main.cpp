#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "image_filters.h"
#include "image_transforms.h"
#include <chrono>
#include <string>
//#include "tiff_io.h"
#include "image_io.h"
#include <algorithm>
#include <sstream>
#include <filesystem>

const char* kernelSource = R"(
// Forward declare the function at the beginning
float findMedian(__local float* radial_window, int size);

__kernel void ringFilter(__global float* input, __global float* output, 
                         int width, int height, int m_rad, int m_azi, 
                         float threshold, int ring_width) {

    int gid = get_global_id(0);  // Global ID for work-item
    int col = gid % width;
    int row = gid / width;

    // Using local memory for faster access
    __local float radial_window[25]; // Adjust size if needed based on m_rad
    int kernel_size = (col < width / 3) ? m_rad / 3 : (col < 2 * width / 3) ? 2 * m_rad / 3 : m_rad;

    // Load the data into local memory for faster access (avoiding global memory reads)
    for (int i = -kernel_size; i <= kernel_size; i++) {
        int adjusted_col = col + i;
        if (adjusted_col < 0) adjusted_col = -adjusted_col;
        else if (adjusted_col >= width) adjusted_col = 2 * width - adjusted_col - 1;
        radial_window[i + kernel_size] = input[row * width + adjusted_col];
    }

    // Ensure all work-items finish loading their data into local memory
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find the median using bubble sort
    float radial_median = findMedian(radial_window, 2 * kernel_size + 1);

    float difference = input[gid] - radial_median;

    // Thresholding step
    if (fabs(difference) > threshold) {
        difference = 0.0f;
    }

    // Azimuthal Mean Filter - can also be parallelized
    float azimuthal_sum = 0.0f;
    int count = 0;
    int azi_kernel_size = (col < width / 3) ? m_azi / 3 : (col < 2 * width / 3) ? 2 * m_azi / 3 : m_azi;

    for (int i = -azi_kernel_size; i <= azi_kernel_size; i++) {
        int adjusted_row = row + i;
        if (adjusted_row < 0) adjusted_row = -adjusted_row;
        else if (adjusted_row >= height) adjusted_row = 2 * height - adjusted_row - 1;

        azimuthal_sum += input[adjusted_row * width + col] - radial_median;
        count++;
    }

    float azimuthal_mean = azimuthal_sum / count;

    // Final output
    output[gid] = azimuthal_mean + radial_median;
}

// Define bubble sort function
void bubbleSort(__local float* arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        for (int j = 0; j < size - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

// Define findMedian function
float findMedian(__local float* radial_window, int size) {
    bubbleSort(radial_window, size);
    return radial_window[size / 2];
}
)";

cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_device_id device;


void setupOpenCL() {
    cl_platform_id platform;
    cl_int err;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    queue = clCreateCommandQueue(context, device, 0, &err);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // Check for build errors and warnings
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        printf("Build log: \n%s\n", log);
        free(log);
    }

    kernel = clCreateKernel(program, "ringFilter", &err);
}


void cleanupOpenCL() {
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}


void doRingFilter(float*** polar_image, int pol_height, int pol_width, float threshold, int m_rad, int m_azi, int ring_width, ImageFilterClass* filter_machine, int verbose, float center_x, float center_y, int width, int height) {
    float* image_block = (float*)calloc(pol_height * pol_width, sizeof(float));
    float** filtered_image = (float**)calloc(pol_height, sizeof(float*));
    filtered_image[0] = image_block;
    for (int i = 1; i < pol_height; i++) {
        filtered_image[i] = filtered_image[i - 1] + pol_width;
    }
    if (verbose == 1) {
        printf("Center X: %f, Center Y: %f\n", center_x, center_y);
        printf("Width: %d, Height: %d\n", width, height);
        printf("Calculated Pol_width: %d, Pol_height: %d\n", pol_width, pol_height);
    }
    //Do radial median filter to get filtered_image
    if (verbose == 1) printf("Performing Radial Filter on polar image... \n");
    clock_t start_median = clock();

    //	filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, 0, pol_height-1, pol_width-1, 'x', (ring_width -1)/2, ring_width, pol_width, pol_height);	
    //	filter_machine->doMedianFilter1D(&filtered_image, polar_image, 0, 0, pol_height-1, pol_width-1, 'x', (ring_width - 1)/2, ring_width, pol_width, pol_height);

    filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, 0, pol_height - 1, pol_width / 3 - 1, 'x', m_rad / 3, ring_width, pol_width, pol_height);
    filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, pol_width / 3, pol_height - 1, 2 * pol_width / 3 - 1, 'x', 2 * m_rad / 3, ring_width, pol_width, pol_height);
    filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, 2 * pol_width / 3, pol_height - 1, pol_width - 1, 'x', m_rad, ring_width, pol_width, pol_height);

    clock_t end_median = clock();
    if (verbose == 1) printf("Time for median filter: %f sec \n", (float(end_median - start_median) / CLOCKS_PER_SEC));

    //subtract filtered image from polar image to get difference image & do last thresholding

    if (verbose == 1) printf("Calculating Difference Image... \n");
    for (int row = 0; row < pol_height; row++) {
        for (int col = 0; col < pol_width; col++) {
            polar_image[0][row][col] -= filtered_image[row][col];
            if (polar_image[0][row][col] > threshold || polar_image[0][row][col] < -threshold) {
                //		if(polar_image[0][row][col] < threshold){
                polar_image[0][row][col] = 0;
            }
        }
    }

    /* Do Azimuthal filter #2 (faster mean, does whole column in one call)
     * using different kernel sizes for the different regions of the image (based on radius)
     */

    if (verbose == 1) printf("Performing Azimuthal mean filter... \n");
    clock_t start_mean = clock();

    filter_machine->doMeanFilterFast1D(&filtered_image, polar_image, 0, 0, pol_height - 1, pol_width / 3 - 1, 'y', m_azi / 3, pol_width, pol_height);
    filter_machine->doMeanFilterFast1D(&filtered_image, polar_image, 0, pol_width / 3, pol_height - 1, 2 * pol_width / 3 - 1, 'y', 2 * m_azi / 3, pol_width, pol_height);
    filter_machine->doMeanFilterFast1D(&filtered_image, polar_image, 0, 2 * pol_width / 3, pol_height - 1, pol_width - 1, 'y', m_azi, pol_width, pol_height);
    /*
        filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, 0, pol_height-1, pol_width/3 -1, 'y', m_azi/3, ring_width, pol_width, pol_height);
        filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, pol_width/3, pol_height-1, 2*pol_width/3 -1, 'y', 2*m_azi/3, ring_width, pol_width, pol_height);
        filter_machine->doMedianFilterFast1D(&filtered_image, polar_image, 0, 2*pol_width/3, pol_height-1, pol_width-1, 'y', m_azi, ring_width, pol_width, pol_height);
    */
    clock_t end_mean = clock();
    if (verbose == 1) printf("Time for mean filtering: %f sec\n", (float(end_mean - start_mean) / CLOCKS_PER_SEC));

    if (verbose == 1) printf("Setting polar image equal to final ring image.. \n");
    //Set "polar_image" to the fully filtered data
    for (int row = 0; row < pol_height; row++) {
        for (int col = 0; col < pol_width; col++) {
            polar_image[0][row][col] = filtered_image[row][col];
        }
    }

    free(filtered_image[0]);
    free(filtered_image);
}

void doRingFilterOpenCL(float*** polar_image, int pol_height, int pol_width,
    float threshold, int m_rad, int m_azi, int ring_width) {
    cl_int err;
    size_t globalSize = pol_width * pol_height;  // Ensure global size matches the number of pixels
    size_t localSize = 256;  // Set a reasonable local size (e.g., 64, 128, 256 depending on the hardware)

    // Ensure localSize is not larger than the maximum allowed
    size_t maxWorkGroupSize;

    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, NULL);
    if (localSize > maxWorkGroupSize) {
        localSize = maxWorkGroupSize;  // Cap the localSize to the maximum supported size
    }

    // Ensure global size is divisible by local size
    if (globalSize % localSize != 0) {
        globalSize = (globalSize / localSize + 1) * localSize;  // Adjust global size
    }

    size_t dataSize = globalSize * sizeof(float);

    // Create buffers
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, dataSize, NULL, &err);
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &err);

    // Write input data to the input buffer
    err = clEnqueueWriteBuffer(queue, inputBuffer, CL_TRUE, 0, dataSize, polar_image[0][0], 0, NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &pol_width);
    clSetKernelArg(kernel, 3, sizeof(int), &pol_height);
    clSetKernelArg(kernel, 4, sizeof(int), &m_rad);
    clSetKernelArg(kernel, 5, sizeof(int), &m_azi);
    clSetKernelArg(kernel, 6, sizeof(float), &threshold);
    clSetKernelArg(kernel, 7, sizeof(int), &ring_width);
    clSetKernelArg(kernel, 8, sizeof(float) * (2 * m_rad + 1), NULL);  // Pass local memory size dynamically

    // Execute the kernel with optimized local size
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error in clEnqueueNDRangeKernel: %d\n", err);
    }

    // Read the output data back to host
    err = clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, dataSize, polar_image[0][0], 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error in clEnqueueReadBuffer: %d\n", err);
    }

    // Release OpenCL memory objects
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
}


std::string getName(const std::string& name_base, int img_num) {
    std::stringstream stream;
    stream << name_base << img_num << ".tif";
    return stream.str();
}

void doRingFilter(float*** polar_image, int pol_height, int pol_width, float threshold, int m_rad, int m_azi, int ring_width, ImageFilterClass* filter_machine, int verbose, float center_x, float center_y, int width, int height);

int main(int argc, char** argv) {
    if (argc != 15) {
        //printf("\nUsage: [Same as original]\n");
        printf("\nUsage: [input path] [output path] [input root] [output root] [first file num] [last file num] [center x] [center y] [ring width] [thresh min] [thresh max] [threshold] [angular min] [verbose]\n");
        return 0;
    }
    setupOpenCL();

    ImageFilterClass* filter_machine = new ImageFilterClass();
    ImageTransformClass* transform_machine = new ImageTransformClass();
    //TiffIO* tiff_io = new TiffIO();
    ImageIO* image_io = new ImageIO();  // Use ImageIO instead of TiffIO

    // Parse command line arguments
    std::string input_path(argv[1]);
    std::string output_path(argv[2]);
    std::string input_base(argv[3]);
    std::string output_base(argv[4]);
    int first_img_num = std::atoi(argv[5]);
    int last_img_num = std::atoi(argv[6]);
    float center_x = std::atof(argv[7]);
    float center_y = std::atof(argv[8]);
    int ring_width = std::atoi(argv[9]);
    float thresh_min = std::atof(argv[10]);
    float thresh_max = std::atof(argv[11]);
    float threshold = std::atof(argv[12]);
    int angular_min = std::atoi(argv[13]);
    int verbose = std::atoi(argv[14]);

    for (int img = first_img_num; img <= last_img_num; img++) {
        auto start = std::chrono::high_resolution_clock::now();
        float** image = nullptr, ** polar_image = nullptr, ** ring_image = nullptr;
        std::string input_name = input_path + input_base + std::to_string(img) + ".png";  // Use PNG
        std::string output_name = output_path + output_base + std::to_string(img) + ".png";  // Use PNG

        if (verbose == 1) printf("\n\nOpening file %s...\n", input_name.c_str());
        int width, height;
        image = image_io->readFloatImage(input_name, &width, &height);  // Read image using OpenCV

        if (!image) {
            fprintf(stderr, "Error: unable to open file %s.\n", input_name.c_str());
            continue;
        }

        if (verbose == 1) printf("Image read. Width: %d, Height: %d\n", width, height);

        if (center_x <= 0 || center_y <= 0 || center_x >= width || center_y >= height) {
            center_x = (width - 1.0) / 2.0;
            center_y = (height - 1.0) / 2.0;
            if (verbose == 1) printf("Using center: (%f, %f)\n", center_x, center_y);
        }

        if (verbose == 1) printf("Performing Polar Transformation...\n");
        int pol_width, pol_height;
        polar_image = transform_machine->polarTransform(image, center_x, center_y, width, height, &pol_width, &pol_height, thresh_max, thresh_min, 1, 1, ring_width);

        if (verbose == 1) printf("Polar transformation complete. Pol_width: %d, Pol_height: %d\n", pol_width, pol_height);

        int m_azi = ceil(float(pol_height) / 360.0) * angular_min;
        int m_rad = 2 * ring_width + 1;

        if (verbose == 1) printf("Starting ring filtering...\n");

        // CPU execution
        auto start_cpu = std::chrono::high_resolution_clock::now();
        doRingFilter(&polar_image, pol_height, pol_width, threshold, m_rad, m_azi, ring_width, filter_machine, verbose, center_x, center_y, width, height);
        auto end_cpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

        // GPU execution
        auto start_gpu = std::chrono::high_resolution_clock::now();
        doRingFilterOpenCL(&polar_image, pol_height, pol_width, threshold, m_rad, m_azi, ring_width);
        auto end_gpu = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

        if (verbose == 1) printf("Ring filtering complete.\n");

        if (verbose == 1) printf("Starting inverse polar transformation...\n");
        ring_image = transform_machine->inversePolarTransform(polar_image, center_x, center_y, pol_width, pol_height, width, height, 1, ring_width);
        if (verbose == 1) printf("Inverse polar transformation complete.\n");

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                image[row][col] -= ring_image[row][col];
            }
        }

        if (verbose == 1) printf("Writing output file: %s\n", output_name.c_str());
        image_io->writeFloatImage(image, output_name, width, height);  // Write image using OpenCV
        if (verbose == 1) printf("Output file written successfully.\n");

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (verbose == 1) {
            printf("Total processing time: %.3f seconds\n", elapsed.count());
            printf("CPU ring filtering time: %.3f seconds\n", cpu_time.count());
            printf("GPU ring filtering time: %.3f seconds\n", gpu_time.count());
            printf("Speedup: %.2fx\n", cpu_time.count() / gpu_time.count());
        }

        // Clean up memory
        free(ring_image[0]);
        free(ring_image);
        free(polar_image[0]);
        free(polar_image);
        free(image[0]);
        free(image);
    }

    if (verbose == 1) printf("Ring Removal completed!\n");

    delete filter_machine;
    delete transform_machine;
    //delete tiff_io;
    delete image_io;

    cleanupOpenCL();
    return 0;
}