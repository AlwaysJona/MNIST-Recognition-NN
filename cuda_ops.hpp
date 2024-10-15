#ifndef CUDA_OPS
#define CUDA_OPS

namespace cuda_ops

{
    // Performing matrix-matrix multiplication: C = A*B where  A is m x k, B is k x n and C is m x n
    __global__ void matrix_mult(const float *A, const float *B, float *C, const int &m, const int &k, const int &n) 
    {
        int rowA = threadIdx.x + blockDim.x*blockIdx.x;
        int colB = threadIdx.y + blockDim.y*blockIdx.y;
        // printf("Row %i Col %i \n", rowA, colB);
        if(rowA < m && colB < n)
        {
            float sum = 0;
            for(int i = 0; i < k; i++)
            {
                sum += A[rowA*k + i]*B[i*n + colB]; 
            }
            //printf("Multiplying row %d and col %d \n", rowA, colB);
            C[rowA*m + colB] = sum;
        }


    }

    // Performing matrix-matrix addition: C = A+B
    __global__ void matrix_add(const float *A, const float *B, float *C, const int &r, const int &c)
    {
        int row = threadIdx.x + blockDim.x*blockIdx.x;
        int col = threadIdx.y + blockDim.y*blockIdx.y;

        if(row < r && col < c)
        {
            //printf("Adding row %d and col %d \n", row, col);
            C[row*c + col] = A[row*c + col] + B[row*c + col];
        }
    }

    // Exponentiating every element of the matrix
    __global__ void matrix_elem_exp(const float *A, float *B, const int& r, const int& c)
    {
        int row = threadIdx.x + blockDim.x*blockIdx.x;
        int col = threadIdx.y + blockDim.y*blockIdx.y;

        if(row < r && col < c)
            B[c*row + col] = exp(A[c*row + col]);
    }

    // Multiplying every element of a matrix by a scalar
    __global__ void scalar_matrix_mult (const float *A, float *B, const int& r, const int& c, const float& scal)
    {
        int row = threadIdx.x + blockDim.x*blockIdx.x;
        int col = threadIdx.y + blockDim.y*blockIdx.y;

        if(row < r && col < c)
            B[c*row + col] = scal*A[c*row + col];
    }

}


#endif
