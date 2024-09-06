#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <vector>
#include "cuda_ops.hpp"
#include "neural.hpp"

TEST_CASE("Testing Cuda matrix operations")
{
    int rowsA = 2, colsA = 3, rowsB = 3, colsB = 2;

    int sizeA = rowsA*colsA;
    
    int sizeB = rowsB*colsB;

    int sizeC = rowsA*colsB;

    int sizeD = sizeA;

    std::vector<float> A, B, C(sizeC), D(sizeD);

    float *d_A, *d_B, *d_C, *d_D;

    for(int i = 0; i < sizeA; i++)
    {               
        A.push_back(i+1);
    }

    for(int i = 0; i < sizeB; i++)
    {               
        B.push_back(i+1);
    }

    cudaMalloc(&d_A, sizeA*sizeof(float));
    cudaMalloc(&d_B, sizeB*sizeof(float));

    cudaMemcpy(d_A, A.data(), sizeA*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB*sizeof(float), cudaMemcpyHostToDevice);

    SUBCASE("Matrix-Matrix Multiplication")
    {
        dim3 threadsperblock(rowsA, colsB);
        cudaMalloc(&d_C, sizeC*sizeof(float));
        cuda_ops::matrix_mult<<<1, threadsperblock >>> (d_A, d_B, d_C, rowsA, colsA, colsB);
        cudaMemcpy(C.data(), d_C, sizeC*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_C);
        CHECK(C[0] == 22);
        CHECK(C[1] == 28);
        CHECK(C[2] == 49);
        CHECK(C[3] == 64);
    }
    SUBCASE("Matrix-Matrix Addition")
    {
        dim3 threadsperblock(rowsA, colsA);
        cudaMalloc(&d_D, sizeD*sizeof(float));
        cuda_ops::matrix_add<<<1, threadsperblock>>> (d_A, d_A, d_D, rowsA, colsA);
        cudaMemcpy(D.data(), d_D, sizeD*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_D);
        for(int i = 0; i < sizeA; i++)
        {
            CHECK(D[i] == 2*A[i]); 
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);

}

TEST_CASE("Testing NN forward propagation")
{
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

    neural::forward_prop(layers, weights, biases, (nodes_per_layer).data(), num_of_layers);

    

}