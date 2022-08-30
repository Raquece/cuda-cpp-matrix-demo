#pragma once

#include <iostream>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.hpp"

namespace cuda_matrix
{
    template<typename T>
    __global__
    void CUDA_matrixMultiply(const T* A, const T* B, T* P, int width, int rows, int cols)
    {
        // Row major format
        int row = blockIdx.y * blockDim.y + threadIdx.y;   
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < rows && col < cols)
        {
            int i = row * cols + col;
            T product = 0;
            for(int k = 0; k < width; k++)
            {
                product += A[row * width + k] * B[k * cols + col];
            }
            P[i] = product;
        }
    }

    template<typename T>
    T* matrixMultiplyFlattened(matrix<T> A, matrix<T> B, int dim_grid_x, int dim_grid_y, int dim_grid_z, int dim_block_x, int dim_block_y, int dim_block_z)
    {
        if (A.cols != B.rows)
        {
            std::cerr << "Matrix A columns (" << A.cols << ") must equal matrix B rows (" << B.rows << ")" << std::endl;
            throw std::invalid_argument("Matrix A columns and matrix B rows are not equal.");
        }

        T* P_flattened, *d_A, *d_B, *d_P;

        matrix<T> P(A.rows, B.cols);

        P_flattened = (T*) malloc(sizeof(T) * P.SIZE);
        cudaMalloc(&d_A, sizeof(T) * A.SIZE);
        cudaMalloc(&d_B, sizeof(T) * B.SIZE);
        cudaMalloc(&d_P, sizeof(T) * P.SIZE);

        cudaMemcpy(d_A, A.DATA, A.SIZE * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.DATA, B.SIZE * sizeof(T), cudaMemcpyHostToDevice);

        CUDA_matrixMultiply<<<dim3(dim_grid_x, dim_grid_y, dim_grid_z), dim3(dim_block_x, dim_block_y, dim_block_z)>>>(d_A, d_B, d_P, A.cols, P.rows, P.cols);

        cudaDeviceSynchronize();

        cudaMemcpy(P_flattened, d_P, P.SIZE * sizeof(T), cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_P);
        free(P);

        return P_flattened;
    }

    template<typename T>
    matrix<T> matrixMultiply(matrix<T> A, matrix<T> B, int dim_grid_x, int dim_grid_y, int dim_grid_z, int dim_block_x, int dim_block_y, int dim_block_z)
    {
        if (A.cols != B.rows)
        {
            std::cerr << "Matrix A columns (" << A.cols << ") must equal matrix B rows (" << B.rows << ")" << std::endl;
            throw std::invalid_argument("Matrix A columns and matrix B rows are not equal.");
        }

        T* d_A, *d_B, *d_P;

        matrix<T> P(A.rows, B.cols);

        cudaMalloc(&d_A, sizeof(T) * A.SIZE);
        cudaMalloc(&d_B, sizeof(T) * B.SIZE);
        cudaMalloc(&d_P, sizeof(T) * P.SIZE);

        cudaMemcpy(d_A, A.DATA, A.SIZE * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, B.DATA, B.SIZE * sizeof(T), cudaMemcpyHostToDevice);

        CUDA_matrixMultiply<<<dim3(dim_grid_x, dim_grid_y, dim_grid_z), dim3(dim_block_x, dim_block_y, dim_block_z)>>>(d_A, d_B, d_P, A.cols, P.rows, P.cols);

        cudaDeviceSynchronize();

        cudaMemcpy(P.DATA, d_P, P.SIZE * sizeof(T), cudaMemcpyDeviceToHost);
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_P);

        return P;
    }
}
