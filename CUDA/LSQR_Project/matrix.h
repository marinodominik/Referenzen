#pragma once
#include <cstdlib>

struct CPUMatrix {
    int height;
    int width;
    int elementSize;
    double *elements;

    int rowSize;
    int columnSize;
    int* csrRow;
    int* csrCol;
};

struct GPUMatrix {
    int height;
    int width;
    int elementSize;
    size_t pitch;
    double *elements;

    int rowSize;
    int* csrRow;
    
    int columnSize;
    int* csrCol;
};
 
//CPU
CPUMatrix matrix_alloc_cpu(int height, int width );
CPUMatrix matrix_alloc_sparse_cpu(int height, int width, int sizeElements, int rowSize, int ColumnSize);
CPUMatrix vector_alloc_sparse_cpu(int height, int width, int sizeElements, int colSize);

void matrix_free_cpu(CPUMatrix &m);
void matrix_free_sparse_cpu(CPUMatrix &m);


//GPU
GPUMatrix matrix_alloc_gpu(int height, int width );
GPUMatrix matrix_alloc_sparse_gpu(int height, int width, int sizeElements, int rowSize, int ColumnSize);
GPUMatrix vector_alloc_sparse_gpu(int height, int width, int sizeElements, int colSize);

void matrix_free_gpu(GPUMatrix &m);
void matrix_free_sparse_gpu(GPUMatrix &m);

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst);
void matrix_download(const GPUMatrix &src, CPUMatrix &dst);

void matrix_upload_cuSparse(const CPUMatrix & src, GPUMatrix & dst);
void matrix_download_cuSparse(const GPUMatrix &src, CPUMatrix & dst);

