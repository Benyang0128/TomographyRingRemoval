#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "image_io.h"

using namespace std;
using namespace cv;

ImageIO::ImageIO() {}
ImageIO::~ImageIO() {}

float** ImageIO::readFloatImage(const string& image_name, int* w_ptr, int* h_ptr)
{
    Mat image = imread(image_name, IMREAD_UNCHANGED);

    if (image.empty()) {
        return nullptr;
    }

    // Handle single-channel and multi-channel images
    if (image.channels() == 1) {
        if (image.type() != CV_32FC1) {
            image.convertTo(image, CV_32FC1, 1.0 / 255.0);
        }
    } else if (image.channels() == 3) {
        // Convert to a single-channel float image (you could choose to handle it differently)
        cvtColor(image, image, COLOR_BGR2GRAY); // Example: Convert to grayscale
        image.convertTo(image, CV_32FC1, 1.0 / 255.0);
    } else {
        return nullptr; // Unsupported format
    }

    *w_ptr = image.cols;
    *h_ptr = image.rows;

    // Allocate float** and populate with data
    float** image_rows = new float*[image.rows];
    for (int i = 0; i < image.rows; i++) {
        image_rows[i] = new float[image.cols];
        memcpy(image_rows[i], image.ptr<float>(i), image.cols * sizeof(float));
    }

    return image_rows;
}

void ImageIO::writeFloatImage(float** image_rows, const string& output_name, int width, int height)
{
    Mat output_image(height, width, CV_32FC1);

    for (int i = 0; i < height; i++) {
        memcpy(output_image.ptr<float>(i), image_rows[i], width * sizeof(float));
    }

    // Normalize the output image for display purposes (if needed)
    Mat output_image_normalized;
    output_image.convertTo(output_image_normalized, CV_8UC1, 255.0); // Scale to 8-bit

    imwrite(output_name, output_image_normalized);
}
