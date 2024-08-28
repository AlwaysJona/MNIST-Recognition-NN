#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <vector>
#include "cuda_ops.hpp"

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