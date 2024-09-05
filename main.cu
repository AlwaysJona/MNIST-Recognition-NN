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

    for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j < 28; j++)
        {
            std::cout << images[0][j + i*28] << " ";
        }
        std::cout << std::endl;
    }

    std::vector<float> first_image = MNIST::format_image(images[0], 28*28, 28);

    for(int i = 0; i < 28; i++)
    {
        for(int j = 0; j < 28; j++)
        {
            std::cout << first_image[j + i*28] << " ";
        }
        std::cout << std::endl;
    }
    
    return 0;
}
// comment
