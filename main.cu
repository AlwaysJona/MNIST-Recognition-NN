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
    
//*******************************************TESTING IN PROGRESS********************************************************* */

    std::vector<float> image = {0, 1, 0,
                                0, 1, 0,
                                0, 1, 0};

    std::vector<float> label = {0,0,1};

    int num_of_layers = 3;
    std::vector<int> nodes_per_layer = {9, 5, 3};

    std::vector<std::vector<float>> weights; // create weights vector

    std::vector<std::vector<float>> biases; // create biases vector

    std::vector<std::vector<float>> layers; // create layers vector

    // initialize layers vector, first layer is image, the rest are zero vectors

    layers.push_back(image);

    for(int i = 1; i < num_of_layers; i++) 
    {
        std::vector<float> temp(nodes_per_layer[i]);
        layers.push_back(temp);
    }

    // initialize weights and biases vectors

    for(int i = 0; i < num_of_layers - 1; i++)
    {
        std::vector<float> w_temp;
        for(int j = 0; j < nodes_per_layer[i]*nodes_per_layer[i+1]; j++)
        {
            w_temp.push_back(0.5);
        }
        weights.push_back(w_temp);

        std::vector<float> b_temp(nodes_per_layer[i]*nodes_per_layer[i+1]);
        biases.push_back(b_temp);
    }

    std::cout << "Last layer before Forward prop: "; // BEFORE

    for(auto x : layers[num_of_layers - 1])
    {
        std::cout << x << " ";
    }

    std::cout << std::endl;

    neural::forward_prop(layers, weights, biases, (nodes_per_layer).data(), num_of_layers);

    std::cout << "Last layer after Forward prop: "; // AFTER

    for(auto x : layers[num_of_layers - 1])
    {
        std::cout << x << " ";
    }

    std::cout << std::endl;


    return 0;
}
// comment
