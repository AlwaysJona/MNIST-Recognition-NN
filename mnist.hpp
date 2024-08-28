#ifndef MNIST_HPP
#define MNIST_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cassert>
namespace MNIST
{
    u_int32_t read_and_bitswap(std::ifstream &file)
    {
        u_int32_t result = 0;
        file.read(reinterpret_cast<char *>(&result), sizeof(result));
        return __builtin_bswap32(result);
    }

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
}

#endif