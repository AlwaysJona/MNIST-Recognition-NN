#include <iostream>
#include "mnist.hpp"
#include <vector>
#include <string>
#include "neural.hpp"
#include "cuda_ops.hpp"

int main()
{
    u_int32_t numIm, numRows, numCol;

    std::vector<std::vector<u_int8_t>> images;

    std::string path = "./../mnist_dataset/t10k-images-idx3-ubyte";

    images = MNIST::read_images(path, numIm, numRows, numCol);

    std::cout<< "Number of Images: " << numIm << ", Number of Rows per Image: " << numRows << ", Number of Columns per Image: " << numCol << std::endl;
    
    int num_of_layers = 3;
    std::vector<int> nodes_per_layer = {28*28, 40, 10};


    return 0;
}