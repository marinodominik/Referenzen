#include <iostream>
#include <cuda_runtime.h>
#include <tuple>
#include "matrix.h"
#include "lsqr.h"
#include "lsqrCUDAcuBlas.h"
#include "lsqrCUDAcuSparse.h"
#include "lsqrCUDAcuSparseKernel.h"
#include "testing.h"
#include "helper.h"
#include "sparseData.h"


void lsqr(const char *pathMatrixA, const char *pathVectorb, const int max_iters) {
    double ebs = 1e-9;
    
    /* <<<< ---------------------- READ DATA ----------------------------- >>>> */
    std::tuple<int, int , double*> A = read_file(pathMatrixA);
    std::tuple<int, int , double*> b = read_file(pathVectorb);

    /* <<<< ---------------- READ DATA IN CPUMATRIX ----------------------- >>>> */
    CPUMatrix cpuMatrixA = matrix_alloc_cpu(std::get<0>(A), std::get<1>(A));
    cpuMatrixA.elements = std::get<2>(A);

    CPUMatrix cpuVector_b = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    cpuVector_b.elements = std::get<2>(b);

    CPUMatrix sparseMatrixA = read_matrix_in_csr(pathMatrixA);


    /* <<<< ---------------- CALCULATE LSQR ONLY WITH cuBLAS-LIBARY ----------------------- >>>> */
    std::cout << "\n\nStarting LSQR using cuBLAS-LIBARY\n" << std::endl;
    CPUMatrix cuBLASResult = cublasLSQR(cpuMatrixA,cpuVector_b, ebs, max_iters);


    //* <<<< ---------------- CALCULATE LSQR ONLY WITH cuSPARSE LIBARY ----------------------- >>>> */
    std::cout << "\n\n\nStarting LSQR using cuSPAPRSE-LIBARY\n" << std::endl;
    CPUMatrix cuSPARSEResult = cusparseLSQR(sparseMatrixA,cpuVector_b,ebs, max_iters);

    CPUMatrix sparseMatrixA2 = read_matrix_in_csr(pathMatrixA);
    CPUMatrix cpuVector_b2 = matrix_alloc_cpu(std::get<0>(b), std::get<1>(b));
    cpuVector_b2.elements = std::get<2>(b);


    kernelCheck(__LINE__);
    /* <<<< ---------------- CALCULATE LSQR ONLY WITH KERNELS ----------------------- >>>> */
    std::cout << "\n\n\nStarting LSQR using kernels\n" << std::endl;
    CPUMatrix kernelResult = sparseLSQR_with_kernels(sparseMatrixA2, cpuVector_b2, max_iters, ebs);


    // Testing
    std::cout << "\n\n\nStarting LSQR using CPU\n" << std::endl;
    compare_lsqr(cpuMatrixA, cpuVector_b, kernelResult, max_iters, ebs);

    /* <<<< ---------------- FREE CPUMATRIX MEMORY ----------------------- >>>>  */
    free(std::get<2>(A));
    free(std::get<2>(b));
    matrix_free_cpu(kernelResult);
    matrix_free_cpu(cuBLASResult);
    matrix_free_cpu(cuSPARSEResult);

}

void printTruncatedVector(CPUMatrix toPrint){
    std::cout<<"[";
    for(int i=0; i<10; i++){
        if(i==toPrint.height){
            std::cout<<"]\n";
            return;
        }
        if(i>0) std::cout<<",";
        std::cout<<toPrint.elements[i]; 
    }
    std::cout<<"...]\n"<<std::endl;
}