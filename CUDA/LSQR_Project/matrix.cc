#include "matrix.h"
#include <cuda_runtime.h>
#include <cstdlib>
#include <iomanip>
#include <iostream>

CPUMatrix matrix_alloc_cpu(int height, int width) {
    CPUMatrix m;
	m.height = height;
    m.width = width;
    m.elements = new double[width * height];
    return m;
}

CPUMatrix matrix_alloc_sparse_cpu(int height, int width, int elementSize, int rowSize, int columnSize) {
	CPUMatrix m;
	m.height = height;
	m.width = width;
	m.elementSize = elementSize;
	m.elements = new double[elementSize];

	m.rowSize = rowSize;
	m.csrRow = new int[rowSize];

	m.columnSize = columnSize;
	m.csrCol = new int[columnSize];
	return m;
}

CPUMatrix vector_alloc_sparse_cpu(int height, int width, int elementSize, int colSize) {
	CPUMatrix m;
	m.height = height;
	m.width = width;
	m.elementSize = elementSize;
	m.elements = new double[elementSize];

	m.columnSize = colSize;
	m.csrCol = new int[colSize];

	return m;
}


void matrix_free_cpu(CPUMatrix &m) {
    delete[] m.elements; 
}

void matrix_free_sparse_cpu(CPUMatrix &m) {
	delete[] m.elements;
	delete[] m.csrRow;
	delete[] m.csrCol;
}


GPUMatrix matrix_alloc_gpu(int height, int width) {
    GPUMatrix Md;
	Md.height = height;
	Md.width = width;
	int size = width * height * sizeof(double);
	cudaError_t err = cudaMalloc(&Md.elements, size);
	return Md;
}

GPUMatrix matrix_alloc_sparse_gpu(int height, int width, int elementSize, int rowSize, int columnSize) {
	GPUMatrix Md;
	Md.height = height;
	Md.width = width;
	Md.elementSize = elementSize;
	cudaError_t err_elements = cudaMalloc(&Md.elements,elementSize*sizeof(double));
	Md.rowSize = rowSize;
	cudaError_t err_csrRow = cudaMalloc(&Md.csrRow, rowSize * sizeof(int));
	Md.columnSize = columnSize;
	cudaError_t err_csrCol = cudaMalloc(&Md.csrCol, columnSize * sizeof(int));
	return Md;
}


GPUMatrix vector_alloc_sparse_gpu(int height, int width, int elementSize, int columnSize) {
	GPUMatrix Md;
	Md.height = height;
	Md.width = width;
	Md.elementSize = elementSize;
	cudaError_t err_elements = cudaMalloc(&Md.elements,elementSize*sizeof(double));

	Md.columnSize = columnSize;
	cudaError_t err_csrCol = cudaMalloc(&Md.csrCol, columnSize * sizeof(int));

	return Md;
}


void matrix_free_gpu(GPUMatrix &m) {
    cudaFree(m.elements);
}


void matrix_free_sparse_gpu(GPUMatrix &m) {
	cudaFree(m.elements);
	cudaFree(m.csrCol);
	cudaFree(m.csrRow);
}


void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	int size = src.height*src.width*sizeof(double);
	cudaMemcpy(dst.elements, src.elements, size, cudaMemcpyHostToDevice);
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	int size = src.height*src.width*sizeof(double);
	cudaMemcpy(dst.elements, src.elements, size, cudaMemcpyDeviceToHost);
}


void matrix_upload_cuSparse(const CPUMatrix & src, GPUMatrix & dst) {

	cudaMemcpy(dst.elements, src.elements, dst.elementSize * sizeof(double), cudaMemcpyHostToDevice);
	//std::cout<<"in upload: elementSize: "<<dst.elementSize<<" height: " << dst.height<< " width: " <<dst.width<<std::endl;
	//for(int i =0; i<dst.height*dst.width; i++)std::cout<< src.elements[i]<<" ";
	//std::cout<<std::endl;
	int size_csrCol = src.columnSize * sizeof(int);
	cudaMemcpy(dst.csrCol, src.csrCol, size_csrCol, cudaMemcpyHostToDevice);

	int size_csrRow = src.rowSize * sizeof(int);
	cudaMemcpy(dst.csrRow, src.csrRow, size_csrRow, cudaMemcpyHostToDevice);
}


void matrix_download_cuSparse(const GPUMatrix &src, CPUMatrix & dst) {
	cudaMemcpy(dst.elements, src.elements, dst.elementSize*sizeof(double), cudaMemcpyDeviceToHost);

	int size_csrCol = src.columnSize * sizeof(int);
	cudaMemcpy(dst.csrCol, src.csrCol, size_csrCol, cudaMemcpyDeviceToHost);

	int size_csrRow = src.rowSize * sizeof(int);
	cudaMemcpy(dst.csrRow, src.csrRow, size_csrRow, cudaMemcpyDeviceToHost);
} 