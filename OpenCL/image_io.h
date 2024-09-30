#ifndef IMAGE_IO_H
#define IMAGE_IO_H

#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <string>

class ImageIO
{
public:
    ImageIO();
    ~ImageIO();

    float** readFloatImage(const std::string& input_name, int* w_ptr, int* h_ptr);
    void writeFloatImage(float** image_rows, const std::string& output_name, int width, int height);
};

#endif  // IMAGE_IO_H
