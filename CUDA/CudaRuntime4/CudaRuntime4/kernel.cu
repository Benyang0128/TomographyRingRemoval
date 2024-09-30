#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <ctime>
#include <chrono>
#include "tiff_io.h"

#define PI 3.14159265359f

using namespace std;

// Helper function to check CUDA calls
void checkCuda(cudaError_t result, const char* msg)
{
    if (result != cudaSuccess)
    {
        fprintf(stderr, "CUDA Runtime Error at %s: %s\n", msg, cudaGetErrorString(result));
        exit(-1);
    }
}

// Function to round a float to the nearest integer
__host__ __device__ int roundFloat(float x)
{
    return static_cast<int>((x >= 0.0f) ? floorf(x + 0.5f) : ceilf(x - 0.5f));
}

__device__ float sign(float x)
{
    return (x > 0) - (x < 0); // Returns 1 if x > 0, -1 if x < 0, and 0 if x == 0
}

// Helper functions for max, min, and abs
template<typename T>
__host__ __device__ T custom_max(T a, T b) {
    return (a > b) ? a : b;
}

template<typename T>
__host__ __device__ T custom_min(T a, T b) {
    return (a < b) ? a : b;
}

template<typename T>
__host__ __device__ T custom_abs(T a) {
    return (a < 0) ? -a : a;
}


// ===========================
// ImageFilterClass Definition
// ===========================
class ImageFilterClass
{
public:
    ImageFilterClass();
    ~ImageFilterClass();

    void doMedianFilterFast1D(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis);
    void doMeanFilterFast1D(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis);
};

// =========================
// ImageFilterClass Methods
// =========================

ImageFilterClass::ImageFilterClass() {}
ImageFilterClass::~ImageFilterClass() {}

// CUDA kernel for median filter along the specified axis
__global__ void medianFilterKernel(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        int window_size = 2 * kernel_rad + 1;
        float window[21]; // Assuming maximum kernel_rad = 10

        int count = 0;
        if (axis == 'x')
        {
            for (int k = -kernel_rad; k <= kernel_rad; k++)
            {
                int idx = x + k;
                idx = max(0, min(idx, width - 1));
                window[count++] = d_image[y * width + idx];
            }
        }
        else // axis == 'y'
        {
            for (int k = -kernel_rad; k <= kernel_rad; k++)
            {
                int idx = y + k;
                idx = max(0, min(idx, height - 1));
                window[count++] = d_image[idx * width + x];
            }
        }

        // Sort the window
        for (int i = 0; i < count - 1; i++)
        {
            for (int j = i + 1; j < count; j++)
            {
                if (window[i] > window[j])
                {
                    float temp = window[i];
                    window[i] = window[j];
                    window[j] = temp;
                }
            }
        }

        // Write the median value to the output image
        d_filtered_image[y * width + x] = window[count / 2];
    }
}

// CUDA kernel for mean filter along the specified axis
__global__ void meanFilterKernel(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float sum = 0.0f;
        int count = 0;

        if (axis == 'x')
        {
            for (int k = -kernel_rad; k <= kernel_rad; k++)
            {
                int idx = x + k;
                if (idx >= 0 && idx < width)
                {
                    sum += d_image[y * width + idx];
                    count++;
                }
            }
        }
        else // axis == 'y'
        {
            for (int k = -kernel_rad; k <= kernel_rad; k++)
            {
                int idx = y + k;
                if (idx >= 0 && idx < height)
                {
                    sum += d_image[idx * width + x];
                    count++;
                }
            }
        }

        d_filtered_image[y * width + x] = sum / count;
    }
}

// Median Filter Function
void ImageFilterClass::doMedianFilterFast1D(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis)
{
    dim3 blockSize(32, 32);
    dim3 gridSize(
        max((width + blockSize.x - 1) / blockSize.x, 1),
        max((height + blockSize.y - 1) / blockSize.y, 1));

    medianFilterKernel << <gridSize, blockSize >> > (d_filtered_image, d_image, width, height, kernel_rad, axis);
    checkCuda(cudaGetLastError(), "Median Filter Kernel");
}

// Mean Filter Function
void ImageFilterClass::doMeanFilterFast1D(float* d_filtered_image, float* d_image, int width, int height, int kernel_rad, char axis)
{
    dim3 blockSize(32, 32);
    dim3 gridSize(
        max((width + blockSize.x - 1) / blockSize.x, 1),
        max((height + blockSize.y - 1) / blockSize.y, 1));

    meanFilterKernel << <gridSize, blockSize >> > (d_filtered_image, d_image, width, height, kernel_rad, axis);
    checkCuda(cudaGetLastError(), "Mean Filter Kernel");
}

// ===========================
// ImageTransformClass Definition
// ===========================
class ImageTransformClass
{
public:
    ImageTransformClass();
    ~ImageTransformClass();

    int findMinDistanceToEdge(float center_x, float center_y, int width, int height);
    float* polarTransform(float* d_image, float center_x, float center_y, int width,
        int height, int* p_pol_width, int* p_pol_height,
        float thresh_max, float thresh_min, int r_scale,
        int ang_scale, int overhang);
    float* inversePolarTransform(float* d_polar_image, float center_x,
        float center_y, int pol_width, int pol_height,
        int width, int height, int r_scale,
        int overhang);
};

// =========================
// ImageTransformClass Methods
// =========================

ImageTransformClass::ImageTransformClass() {}
ImageTransformClass::~ImageTransformClass() {}

int ImageTransformClass::findMinDistanceToEdge(float center_x, float center_y, int width, int height)
{
    int dist[4];
    dist[0] = static_cast<int>(center_x);
    dist[1] = static_cast<int>(center_y);
    dist[2] = width - static_cast<int>(center_x);
    dist[3] = height - static_cast<int>(center_y);
    return *std::min_element(dist, dist + 4);
}

// CUDA kernel for polar transformation
__global__ void polarTransformKernel(float* d_polar_image, float* d_image,
    float center_x, float center_y, int pol_width,
    int pol_height, int width, int height,
    float thresh_max, float thresh_min, int r_scale)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int theta_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < pol_width && theta_idx < pol_height)
    {
        // Compute polar coordinates (r, theta)
        float theta = theta_idx * 2.0f * PI / pol_height;
        float x_f = center_x + (r / static_cast<float>(r_scale)) * cosf(theta);
        float y_f = center_y + (r / static_cast<float>(r_scale)) * sinf(theta);

        // Bilinear interpolation setup
        int x0 = floorf(x_f);  // lower bound in x
        int y0 = floorf(y_f);  // lower bound in y
        int x1 = x0 + 1;       // upper bound in x
        int y1 = y0 + 1;       // upper bound in y

        // Check if bounds are within the image dimensions
        if (x0 >= 0 && x1 < width && y0 >= 0 && y1 < height)
        {
            // Calculate fractional parts
            float x_frac = x_f - x0;
            float y_frac = y_f - y0;

            // Get pixel values at four corners
            float val00 = d_image[y0 * width + x0]; // top-left
            float val01 = d_image[y1 * width + x0]; // bottom-left
            float val10 = d_image[y0 * width + x1]; // top-right
            float val11 = d_image[y1 * width + x1]; // bottom-right

            // Bilinear interpolation formula
            float val0 = (1.0f - x_frac) * val00 + x_frac * val10;
            float val1 = (1.0f - x_frac) * val01 + x_frac * val11;
            float interpolated_value = (1.0f - y_frac) * val0 + y_frac * val1;

            // Apply thresholds to the interpolated value
            interpolated_value = fminf(fmaxf(interpolated_value, thresh_min), thresh_max);

            // Write the result to the polar image
            d_polar_image[theta_idx * pol_width + r] = interpolated_value;
        }
        else
        {
            // If outside bounds, set polar image pixel to 0
            d_polar_image[theta_idx * pol_width + r] = 0.0f;
        }
    }
}


// Polar Transform Function
float* ImageTransformClass::polarTransform(float* d_image, float center_x, float center_y, int width,
    int height, int* p_pol_width, int* p_pol_height,
    float thresh_max, float thresh_min, int r_scale,
    int ang_scale, int overhang)
{
    int max_r = findMinDistanceToEdge(center_x, center_y, width, height) + overhang;
    if (max_r <= 0)
        max_r = 1;

    int pol_width = r_scale * max_r;
    int pol_height = ang_scale * 360; // Assuming ang_scale is samples per degree
    *p_pol_width = pol_width;
    *p_pol_height = pol_height;

    size_t polar_image_size = pol_width * pol_height * sizeof(float);
    float* d_polar_image;
    checkCuda(cudaMalloc(&d_polar_image, polar_image_size), "Allocating d_polar_image");

    dim3 blockSize(32, 32);
    dim3 gridSize(
        max((pol_width + blockSize.x - 1) / blockSize.x, 1),
        max((pol_height + blockSize.y - 1) / blockSize.y, 1)
    );

    polarTransformKernel << <gridSize, blockSize >> > (d_polar_image, d_image, center_x, center_y,
        pol_width, pol_height, width, height,
        thresh_max, thresh_min, r_scale);
    checkCuda(cudaGetLastError(), "Polar Transform Kernel");

    return d_polar_image;
}

// CUDA kernel for inverse polar transformation
__global__ void inversePolarTransformKernel(float* d_cart_image, float* d_polar_image,
    float center_x, float center_y, int pol_width,
    int pol_height, int width, int height, int r_scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        float dx = x - center_x;
        float dy = y - center_y;
        float r = sqrtf(dx * dx + dy * dy) * r_scale;
        float theta = atan2f(dy, dx);
        if (theta < 0)
            theta += 2.0f * PI;

        int r_idx = static_cast<int>(r);
        int theta_idx = static_cast<int>(theta * pol_height / (2.0f * PI));

        if (r_idx >= 0 && r_idx < pol_width && theta_idx >= 0 && theta_idx < pol_height)
        {
            d_cart_image[y * width + x] = d_polar_image[theta_idx * pol_width + r_idx];
        }
        else
        {
            d_cart_image[y * width + x] = 0.0f;
        }
    }
}

// Inverse Polar Transform Function
float* ImageTransformClass::inversePolarTransform(float* d_polar_image, float center_x,
    float center_y, int pol_width, int pol_height,
    int width, int height, int r_scale,
    int overhang)
{
    size_t cart_image_size = width * height * sizeof(float);
    float* d_cart_image;
    checkCuda(cudaMalloc(&d_cart_image, cart_image_size), "Allocating d_cart_image");

    dim3 blockSize(32, 32);
    dim3 gridSize(
        max((width + blockSize.x - 1) / blockSize.x, 1),
        max((height + blockSize.y - 1) / blockSize.y, 1)
    );

    inversePolarTransformKernel << <gridSize, blockSize >> > (d_cart_image, d_polar_image, center_x,
        center_y, pol_width, pol_height,
        width, height, r_scale);
    checkCuda(cudaGetLastError(), "Inverse Polar Transform Kernel");

    return d_cart_image;
}

// ======================
// Ring Removal Kernels
// ======================

// Kernel to calculate difference and apply thresholding
__global__ void differenceAndThresholdKernel(float* d_diff, float* d_orig, float* d_med, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        float diff = d_orig[idx] - d_med[idx];
        if (fabsf(diff) > threshold)
        {
            d_diff[idx] = diff - sign(diff) * (fabsf(diff) - threshold) * 0.5f;  // Gradual reduction
        }
        else
        {
            d_diff[idx] = diff;
        }
    }
}

// Kernel to subtract images
__global__ void subtractKernel(float* d_result, float* d_image1, float* d_image2, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height)
    {
        int idx = y * width + x;
        d_result[idx] = d_image1[idx] - d_image2[idx];
    }
}

// ======================
// Ring Removal Function
// ======================

void doRingFilter(float* d_image, int width, int height, float threshold, int m_rad, int m_azi, int ring_width, ImageFilterClass* filter_machine, ImageTransformClass* transform_machine)
{
    int pol_width, pol_height;
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    int r_scale = 1;
    int ang_scale = 1;
    float thresh_max = 1e6f;
    float thresh_min = -1e6f;

    cudaEvent_t start, stop;
    float polarTransformTime, medianFilterTime, differenceThresholdTime, meanFilterTime, inversePolarTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Polar transformation timing
    cudaEventRecord(start);
    float* d_polar_image = transform_machine->polarTransform(d_image, center_x, center_y, width, height,
        &pol_width, &pol_height, thresh_max, thresh_min,
        r_scale, ang_scale, ring_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&polarTransformTime, start, stop);

    // Allocate memory for intermediate results
    float* d_median_filtered, * d_difference, * d_mean_filtered;
    size_t pol_image_size = pol_width * pol_height * sizeof(float);
    checkCuda(cudaMalloc(&d_median_filtered, pol_image_size), "Allocating d_median_filtered");
    checkCuda(cudaMalloc(&d_difference, pol_image_size), "Allocating d_difference");
    checkCuda(cudaMalloc(&d_mean_filtered, pol_image_size), "Allocating d_mean_filtered");

    // Radial median filter timing
    cudaEventRecord(start);
    filter_machine->doMedianFilterFast1D(d_median_filtered, d_polar_image, pol_width, pol_height, m_rad, 'x');
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&medianFilterTime, start, stop);

    // Difference and threshold timing
    cudaEventRecord(start);
    dim3 blockSize(32, 32);
    dim3 gridSize(
        max((pol_width + blockSize.x - 1) / blockSize.x, 1),
        max((pol_height + blockSize.y - 1) / blockSize.y, 1)
    );
    differenceAndThresholdKernel << <gridSize, blockSize >> > (d_difference, d_polar_image, d_median_filtered, pol_width, pol_height, threshold);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&differenceThresholdTime, start, stop);

    // Azimuthal mean filter timing
    cudaEventRecord(start);
    filter_machine->doMeanFilterFast1D(d_mean_filtered, d_difference, pol_width, pol_height, m_azi, 'y');
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&meanFilterTime, start, stop);

    // Inverse polar transformation timing
    cudaEventRecord(start);
    float* d_ring_image = transform_machine->inversePolarTransform(d_mean_filtered, center_x, center_y,
        pol_width, pol_height, width, height,
        r_scale, ring_width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&inversePolarTime, start, stop);

    // Subtract ring image from original image
    dim3 cartBlockSize(32, 32);
    dim3 cartGridSize(
        max((width + cartBlockSize.x - 1) / cartBlockSize.x, 1),
        max((height + cartBlockSize.y - 1) / cartBlockSize.y, 1)
    );
    subtractKernel << <cartGridSize, cartBlockSize >> > (d_image, d_image, d_ring_image, width, height);

    // Print timing results
    printf("Polar Transformation Time: %.3f ms\n", polarTransformTime);
    printf("Radial Median Filter Time: %.3f ms\n", medianFilterTime);
    printf("Difference and Threshold Time: %.3f ms\n", differenceThresholdTime);
    printf("Azimuthal Mean Filter Time: %.3f ms\n", meanFilterTime);
    printf("Inverse Polar Transformation Time: %.3f ms\n", inversePolarTime);

    // Free allocated memory
    cudaFree(d_polar_image);
    cudaFree(d_median_filtered);
    cudaFree(d_difference);
    cudaFree(d_mean_filtered);
    cudaFree(d_ring_image);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ======================
// Main Function
// ======================

string getName(string name_base, int img_num)
{
    stringstream stream;
    stream << name_base << img_num << ".png";
    return stream.str();
}

// Add CPU versions of the functions
void cpuPolarTransform(float* cart_image, float* polar_image, int width, int height, int pol_width, int pol_height, float center_x, float center_y, int r_scale)
{
    for (int r = 0; r < pol_width; r++) {
        for (int theta_idx = 0; theta_idx < pol_height; theta_idx++) {
            float theta = theta_idx * 2.0f * PI / pol_height;
            float x_f = center_x + (r / static_cast<float>(r_scale)) * cosf(theta);
            float y_f = center_y + (r / static_cast<float>(r_scale)) * sinf(theta);

            int x = roundFloat(x_f);
            int y = roundFloat(y_f);

            if (x >= 0 && x < width && y >= 0 && y < height) {
                polar_image[theta_idx * pol_width + r] = cart_image[y * width + x];
            }
            else {
                polar_image[theta_idx * pol_width + r] = 0.0f;
            }
        }
    }
}

void cpuMedianFilter1D(float* input, float* output, int width, int height, int kernel_rad, char axis)
{
    float* window = new float[2 * kernel_rad + 1];

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int count = 0;
            if (axis == 'x') {
                for (int k = -kernel_rad; k <= kernel_rad; k++) {
                    int idx = x + k;
                    idx = (idx < 0) ? 0 : (idx >= width ? width - 1 : idx);
                    window[count++] = input[y * width + idx];
                }
            }
            else {
                for (int k = -kernel_rad; k <= kernel_rad; k++) {
                    int idx = y + k;
                    idx = (idx < 0) ? 0 : (idx >= height ? height - 1 : idx);
                    window[count++] = input[idx * width + x];
                }
            }

            // Simple bubble sort for median
            for (int i = 0; i < count - 1; i++) {
                for (int j = 0; j < count - i - 1; j++) {
                    if (window[j] > window[j + 1]) {
                        float temp = window[j];
                        window[j] = window[j + 1];
                        window[j + 1] = temp;
                    }
                }
            }
            output[y * width + x] = window[count / 2];
        }
    }

    delete[] window;
}

void cpuMeanFilter1D(float* input, float* output, int width, int height, int kernel_rad, char axis)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            int count = 0;

            if (axis == 'x') {
                for (int k = -kernel_rad; k <= kernel_rad; k++) {
                    int idx = x + k;
                    if (idx >= 0 && idx < width) {
                        sum += input[y * width + idx];
                        count++;
                    }
                }
            }
            else {
                for (int k = -kernel_rad; k <= kernel_rad; k++) {
                    int idx = y + k;
                    if (idx >= 0 && idx < height) {
                        sum += input[idx * width + x];
                        count++;
                    }
                }
            }

            output[y * width + x] = sum / count;
        }
    }
}

void cpuInversePolarTransform(float* polar_image, float* cart_image, int pol_width, int pol_height, int width, int height, float center_x, float center_y, int r_scale)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float dx = x - center_x;
            float dy = y - center_y;
            float r = sqrtf(dx * dx + dy * dy) * r_scale;
            float theta = atan2f(dy, dx);
            if (theta < 0)
                theta += 2.0f * PI;

            int r_idx = static_cast<int>(r);
            int theta_idx = static_cast<int>(theta * pol_height / (2.0f * PI));

            if (r_idx >= 0 && r_idx < pol_width && theta_idx >= 0 && theta_idx < pol_height) {
                cart_image[y * width + x] = polar_image[theta_idx * pol_width + r_idx];
            }
            else {
                cart_image[y * width + x] = 0.0f;
            }
        }
    }
}

void cpuRingFilter(float* image, int width, int height, float threshold, int m_rad, int m_azi, int ring_width)
{
    int pol_width, pol_height;
    float center_x = width / 2.0f;
    float center_y = height / 2.0f;
    int r_scale = 1;
    int ang_scale = 1;

    pol_width = r_scale * custom_min(custom_min(static_cast<int>(center_x), static_cast<int>(center_y)),
        custom_min(static_cast<int>(width - center_x), static_cast<int>(height - center_y)));
    pol_height = ang_scale * 360;

    float* polar_image = new float[pol_width * pol_height];
    float* median_filtered = new float[pol_width * pol_height];
    float* difference = new float[pol_width * pol_height];
    float* mean_filtered = new float[pol_width * pol_height];

    cpuPolarTransform(image, polar_image, width, height, pol_width, pol_height, center_x, center_y, r_scale);
    cpuMedianFilter1D(polar_image, median_filtered, pol_width, pol_height, m_rad, 'x');

    for (int i = 0; i < pol_width * pol_height; i++) {
        float diff = polar_image[i] - median_filtered[i];
        difference[i] = (custom_abs(diff) > threshold) ? 0.0f : diff;
    }

    cpuMeanFilter1D(difference, mean_filtered, pol_width, pol_height, m_azi, 'y');

    float* ring_image = new float[width * height];
    cpuInversePolarTransform(mean_filtered, ring_image, pol_width, pol_height, width, height, center_x, center_y, r_scale);

    for (int i = 0; i < width * height; i++) {
        image[i] -= ring_image[i];
    }

    delete[] polar_image;
    delete[] median_filtered;
    delete[] difference;
    delete[] mean_filtered;
    delete[] ring_image;
}

int main(int argc, char** argv)
{
    if (argc != 15)
    {
        printf("\nUsage:\n\nring_remover_recon [input path] [output path] [input root] [output root] [first file num] [last file num] [center x y] [max ring width] [thresh min max] [ring threshold] [angular min] [verbose]\n\n");
        return 0;
    }
    else
    {
        // Remove TIFF-specific error handlers
        int first_img_num = atoi(argv[5]);
        int last_img_num = atoi(argv[6]);
        float center_x = atof(argv[7]);
        float center_y = atof(argv[8]);
        int ring_width = atoi(argv[9]);
        float thresh_min = atof(argv[10]);
        float thresh_max = atof(argv[11]);
        float threshold = atof(argv[12]);
        int angular_min = atoi(argv[13]);
        int verbose = atoi(argv[14]);

        string input_path(argv[1]);
        string output_path(argv[2]);
        string input_base(argv[3]);
        string output_base(argv[4]);

        if (input_path.back() != '/' && input_path.back() != '\\')
            input_path += '/';
        if (output_path.back() != '/' && output_path.back() != '\\')
            output_path += '/';

        ImageFilterClass filter_machine;
        ImageTransformClass transform_machine;
        ImageIO image_io;

        for (int img = first_img_num; img <= last_img_num; img++)
        {
            string input_name = getName(input_base, img);
            string output_name = getName(output_base, img);

            printf("Opening file %s...\n", (input_path + input_name).c_str());
            int width = 0, height = 0;
            float** h_image_rows = image_io.readFloatImage(input_path + input_name, &width, &height);
            if (!h_image_rows)
            {
                fprintf(stderr, "Error: unable to open file %s.\n", (input_path + input_name).c_str());
                continue;
            }
            printf("Image read. Width: %d, Height: %d\n", width, height);
            printf("Using center: (%f, %f)\n", center_x, center_y);

            // Convert float** to float* for GPU processing
            float* h_image = new float[width * height];
            for (int i = 0; i < height; i++)
            {
                memcpy(h_image + i * width, h_image_rows[i], width * sizeof(float));
            }

            // CPU Ring Removal
            float* cpu_result = new float[width * height];
            std::memcpy(cpu_result, h_image, width * height * sizeof(float));

            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpuRingFilter(cpu_result, width, height, threshold, ring_width, angular_min, ring_width);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

            // GPU Ring Removal
            float* d_image;
            size_t image_size = width * height * sizeof(float);
            checkCuda(cudaMalloc(&d_image, image_size), "Allocating d_image");
            checkCuda(cudaMemcpy(d_image, h_image, image_size, cudaMemcpyHostToDevice), "Copying h_image to d_image");

            auto gpu_start = std::chrono::high_resolution_clock::now();
            doRingFilter(d_image, width, height, threshold, ring_width, angular_min, ring_width, &filter_machine, &transform_machine);
            auto gpu_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> gpu_duration = gpu_end - gpu_start;

            float* gpu_result = new float[width * height];
            checkCuda(cudaMemcpy(gpu_result, d_image, image_size, cudaMemcpyDeviceToHost), "Copying result to host");

            float** result_rows = new float* [height];
            for (int i = 0; i < height; i++)
            {
                result_rows[i] = new float[width];
                memcpy(result_rows[i], gpu_result + i * width, width * sizeof(float));
            }

            // Write out corrected image
            printf("Writing output file: %s\n", (output_path + output_name).c_str());
            image_io.writeFloatImage(result_rows, output_path + output_name, width, height);
            printf("Output file written successfully.\n");

            printf("CPU ring filtering time: %.3f seconds\n", cpu_duration.count());
            printf("GPU ring filtering time: %.3f seconds\n", gpu_duration.count());
            printf("Speedup: %.2fx\n", cpu_duration.count() / gpu_duration.count());

            // Free memory
            cudaFree(d_image);
            for (int i = 0; i < height; i++)
            {
                delete[] h_image_rows[i];
                delete[] result_rows[i];
            }
            delete[] h_image_rows;
            delete[] result_rows;
            delete[] h_image;
            delete[] gpu_result;
        }

        printf("Ring Removal completed!\n");
        return 0;
    }
}