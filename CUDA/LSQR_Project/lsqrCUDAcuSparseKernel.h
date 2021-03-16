#pragma once
#include <iostream>
#include "matrix.h"
/*#include "lsqrCUDAcuBlas.h"*/
#include "math.h"
#include "float.h"


double getNorm2(const GPUMatrix denseVector);
void get_add_subtract_vector(GPUMatrix denseA, GPUMatrix denseB, bool operation);
void multiply_scalar_vector(const GPUMatrix vector, const double scalar);
void get_csr_matrix_vector_multiplication(const GPUMatrix matrix, const GPUMatrix vector, GPUMatrix result);
//GPUMatrix get_csr_matrix_vector_multiplication_sh(const GPUMatrix A_sparse, const GPUMatrix b_dense, GPUMatrix result);

void kernelCheck(int line);
void printValuesKernel(GPUMatrix x, const char *name); 

void printVectorKernel(int iteration,GPUMatrix x, const char* name);

GPUMatrix transpose_matrix(GPUMatrix A);

CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const int max_iters, const double ebs);
GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const int max_iters, const double ebs);

inline unsigned int div_up(unsigned int numerator, unsigned int denominator);