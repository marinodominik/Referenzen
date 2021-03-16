#pragma once

#pragma once
#include "matrix.h"
#include <cublas_v2.h>
#include "lsqrCUDAcuSparseKernel.h"
#include "lsqr.h"
#include "matrix.h"
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <iostream>


CPUMatrix cusparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs,int max_iterations);
CPUMatrix cusparseLSQR_aux(const CPUMatrix &A, const GPUMatrix &VECb,GPUMatrix &VECu,GPUMatrix &VECv,GPUMatrix &VECw,GPUMatrix &VECx,GPUMatrix &tempVector,double ebs,int max_iterations);
void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A,cusparseDnVecDescr_t u, cusparseDnVecDescr_t v,cusparseDnVecDescr_t x, cusparseDnVecDescr_t tempVector);
void cuSPARSECheck(int line);
void printVector(int iteration,GPUMatrix x, const char* name);
void initGPUVectors(const CPUMatrix &b, GPUMatrix &u,GPUMatrix &v, GPUMatrix& x, GPUMatrix &w, GPUMatrix &GPUb, GPUMatrix &tempVector);
void cleanGPUVectors(GPUMatrix& u,GPUMatrix &v, GPUMatrix &x, GPUMatrix &w, GPUMatrix &GPUb, GPUMatrix &tempVector);