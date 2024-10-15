#ifndef MNIST_HPP
#define MNIST_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
#include <cuda.h>

namespace MNIST
{   // used to properly reading binary numbers from the mnist dataset
    u_int32_t read_and_bitswap(std::ifstream &file)
    {
        u_int32_t result = 0;
        file.read(reinterpret_cast<char *>(&result), sizeof(result));
        return __builtin_bswap32(result);
    }

    // used to read the images and store them in an array, returns 2D array array of integers, the images
    std::vector<std::vector<u_int8_t>> read_images(std::string path, u_int32_t &numIm, u_int32_t &numRows, u_int32_t &numCol)
    {
        std::ifstream file(path, std::ios::binary);
        if(!file.is_open())
        {
            std::cerr << "Unable to Open File";
            exit(EXIT_FAILURE);
        }

        u_int32_t magic = read_and_bitswap(file);
        assert(magic == 2051 && "Invalid Magic Number for Image File");

        numIm = read_and_bitswap(file);
        numRows = read_and_bitswap(file);
        numCol = read_and_bitswap(file);

        std::vector<std::vector<u_int8_t>> images(numIm, std::vector<u_int8_t>(numCol * numRows));
        for(int i = 0; i < numIm; i++ )
        {
            file.read(reinterpret_cast<char *> (images[i].data()), numCol * numRows); 
        }

        return images;
    }

    // reading the list of labels, return array of integers, the labels for each integer
    std::vector<u_int8_t> read_labels(std::string path)
    {
        std::ifstream file(path, std::ios::binary);
        if(!file.is_open())
        {
            std::cerr << "Unable to Open File";
            exit(EXIT_FAILURE);
        }

        u_int32_t magic = read_and_bitswap(file);

        assert(magic == 2049 && "Invalid Magic Number for Label File");

        u_int32_t numLabels = read_and_bitswap(file);
        std::vector<u_int8_t> labels(numLabels);

        for(int l = 0; l < numLabels; l++)
        {
            file.read(reinterpret_cast<char *>(labels.data()), numLabels);
        }
        
        return labels;
    }

    // device function, used to format the pixels in the images to floating point values between 0 and 1
    __global__
    void gpu_format_image(const u_int8_t* image, float* formatted, const int size)
    {
        int row = threadIdx.x + blockDim.x*blockIdx.x;

        if ( row < size*size)
        {
            formatted[row] = (float)image[row]/(float)255;
        }

    }

    // host funtion to format the images, returns array of formatted pixels, formats 1 image
    std::vector<float> format_image(std::vector<u_int8_t> image, const int num_of_pixles, const int size)
    {
        std::vector<float> formatted(num_of_pixles);
            
        u_int8_t* d_image;
        float* d_formatted;

        cudaMalloc(&d_image, sizeof(u_int8_t)*num_of_pixles);
        cudaMalloc(&d_formatted, sizeof(float)*num_of_pixles);

        cudaMemcpy(d_image, image.data(), sizeof(u_int8_t)*num_of_pixles, cudaMemcpyHostToDevice);
        cudaMemcpy(d_formatted, formatted.data(), sizeof(u_int8_t)*num_of_pixles, cudaMemcpyHostToDevice);

        dim3 blockSize(64,1); // threads per block
        dim3 gridSize((int)(num_of_pixles + blockSize.x - 1)/blockSize.x,1); // blocks in a grid
        gpu_format_image<<<gridSize,blockSize>>>(d_image, d_formatted, size);

        cudaMemcpy(formatted.data(), d_formatted, sizeof(float)*num_of_pixles, cudaMemcpyDeviceToHost);

        cudaFree(d_image);
        cudaFree(d_formatted);

        return formatted;

    } 

    
}

#endif
