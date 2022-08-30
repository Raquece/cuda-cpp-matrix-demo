#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <functional>

#include "cuda_matrix.cu"
#include "matrix.hpp"

using namespace std;
using namespace cuda_matrix;

int main(void)
{
    int A_rows = 1 << 8;
    int A_cols = 1 << 10;
    int B_rows = A_cols;
    int B_cols = 1 << 12;

    matrix<float> A(A_rows, A_cols);
    matrix<float> B(B_rows, B_cols);
    A.init(4.0f);
    B.init(2.0f);
    
    matrix<float> P = cuda_matrix::matrixMultiply(A, B, B.cols / 32, A.rows / 32, 1, 32, 32, 1);

    A.free();
    B.free();
    P.free();
}
