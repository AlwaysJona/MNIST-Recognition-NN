#ifndef NEURAL_HPP
#define NEURAL_HPP

#include <vector>
#include <cuda.h>
#include <stdio.h>
#include "cuda_ops.hpp"

namespace neural
{
    // ReLU on every element of a matrix
    __global__
    void ReLU(const float *A, float *B, const int& r, const int& c)
    {
        int row = threadIdx.x + blockDim.x*blockIdx.x;
        int col = threadIdx.y + blockDim.y*blockIdx.y;

        if(row < r && col < c)
            B[c*row + col] = (A[c * row + col] > 0 ? A[c * row + col] : 0);
    }

    // Applying softmax on every element of a matrix
    std::vector<float> Softmax(const float *A, const int& row, const int& col, dim3 grid_shape, dim3 block_shape)
    {
	int num_of_nodes = row*col;

	float* d_A;
	float* d_B;
	float* d_C;
	cudaMalloc(&d_A, sizeof(float)*num_of_nodes);
	cudaMalloc(&d_B, sizeof(float)*num_of_nodes);
	cudaMalloc(&d_C, sizeof(float)*num_of_nodes);

	cudaMemcpy(d_A, A, sizeof(float)*num_of_nodes, cudaMemcpyHostToDevice);
        cuda_ops::matrix_elem_exp<<<grid_shape, block_shape>>> (d_A, d_B, row, col);

	printf("Succesfully exponentiated elements in matrix \n");

	std::vector<float> B(num_of_nodes);
	cudaMemcpy(B.data(), d_B, sizeof(float)*num_of_nodes, cudaMemcpyDeviceToHost);

        float sum = 0;
        for (int i = 0; i < num_of_nodes; i++)
        {
            sum += B[i];
	    printf("Summing elemnts: iteration %i \n", i);
        }

        sum = 1/sum;

        cuda_ops::scalar_matrix_mult<<<grid_shape, block_shape>>> (d_B, d_C, row, col, sum);

	std::vector<float> C(num_of_nodes);

	cudaMemcpy(C.data(), d_C, sizeof(float)*num_of_nodes, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return C;
    }

    // Initializing weights of each layer
    void weight_init (float **weights, float** biases, const int& num_of_layers, const int *nodes_per_layer)
    {
        for (int i = 1; i < num_of_layers; i++) // layer[0] is input layer and layer[num_of_layers] is the last layer to which softmax needs to be applied
        {
            int size = nodes_per_layer[i] * nodes_per_layer[i-1];
            float stddev = sqrt(1/nodes_per_layer[i]);
            for(int j = 0; j < size; j++)           // the weights of a fully connected NN are martices of dimensions: nodes_per_layer[i-1] x nodes_per_layer[i]
            {
                weights[i][j] = rand() % ((int) (2* stddev * 100000));
                weights[i][j] -= stddev * 100000;
                weights[i][j] = (float)(weights[i][j]/100000);
            }
            for(int j = 0; j < nodes_per_layer[i]; j++)
            {
                biases[i][j] = 0;
            }
        }
    }

    // calculate average cross entropy loss from the prob distribution array resulting from applying softmax to the last layer
    float cross_entropy_loss(const float* prob, const float* truth, const int& size)
    {
        float loss = 0; 
        for(int i = 0; i < size; i++)
        {
            loss += truth[i] * log(prob[i]) + (1 - truth[i])*log(1 - prob[i]);
        }

        loss = loss/size;
        return loss;
    }
    // forward propagation of the layers starting from the input layer
    void forward_prop(std::vector<std::vector<float>>& layers, const std::vector<std::vector<float>> weights, const std::vector<std::vector<float>> biases, const int* nodes_per_layer, const int& num_of_layers)
    {
        for(int i = 0; i < num_of_layers - 1; i++)
        {
            float* d_layer;
            float* d_weights;
            float* d_biases;
            float* d_temp;
            
            cudaMalloc(&d_layer, sizeof(float)*nodes_per_layer[i]);
            cudaMalloc(&d_weights, sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1]);
            cudaMalloc(&d_biases, sizeof(float)*nodes_per_layer[i]);
            cudaMalloc(&d_temp, sizeof(float)*nodes_per_layer[i+1]);

            cudaMemcpy(d_layer, layers[i].data(), sizeof(float)*nodes_per_layer[i], cudaMemcpyHostToDevice);
            cudaMemcpy(d_weights, weights[i].data(), sizeof(float)*nodes_per_layer[i]*nodes_per_layer[i+1], cudaMemcpyHostToDevice);
            cudaMemcpy(d_biases, biases[i].data(), sizeof(float)*nodes_per_layer[i], cudaMemcpyHostToDevice);

            // dim3 threadsPerBlock(64, 1);  // 64 threads in the x-dimension
            // dim3 numBlocks(1, (nodes_per_layer[i+1] + threadsPerBlock.x - 1) / threadsPerBlock.x);
            dim3 threadsPerBlock(nodes_per_layer[i], nodes_per_layer[i+1]);
            dim3 numBlocks(1, 1);

            cuda_ops::matrix_mult<<<numBlocks,threadsPerBlock>>>(d_layer, d_weights, d_temp, 1, nodes_per_layer[i], nodes_per_layer[i+1]); // store layers as row vectors

            std::cout << "Iteration: " << i << ", Multiplication Done" << std::endl;

            cuda_ops::matrix_add<<<numBlocks,threadsPerBlock>>>(d_layer, d_biases, d_layer, 1, nodes_per_layer[i+1]);

            std::cout << "Iteration: " << i << ", Addition Done" << std::endl;

            cudaMemcpy(layers[i+1].data(), d_temp, sizeof(float)*nodes_per_layer[i+1], cudaMemcpyDeviceToHost);

            cudaFree(d_layer);
            cudaFree(d_weights);
            cudaFree(d_biases);
            cudaFree(d_temp);

        }

        std::cout << "All iterations completed" << std::endl;

        dim3 threadsPerBlock(64, 1);  // 64 threads in the x-dimension
        dim3 numBlocks((nodes_per_layer[num_of_layers - 1] + threadsPerBlock.x - 1) / threadsPerBlock.x, 1);

        // applying softmax to the last layer

	std::cout << "Printing last Layer after going through NN but before Softmax" << std::endl;

	for(auto x : layers[num_of_layers - 1])
	{
	    std::cout << x << " ";
	}

	std::cout << std::endl;

	layers[num_of_layers - 1] = Softmax(layers[num_of_layers - 1].data(), nodes_per_layer[num_of_layers - 1], 1, numBlocks, threadsPerBlock);

        std::cout << "Softmax Applied" << std::endl;

	std::cout << "Printing Last Layer After Softmax" << std::endl;
	
	for(auto x : layers[num_of_layers - 1])
	{
	   std::cout << x << " ";
	}
	std::cout << std::endl;

    }
}
#endif
