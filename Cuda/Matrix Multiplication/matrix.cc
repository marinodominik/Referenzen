#include <iostream>
#include <iomanip>
#include <cstdlib>

#include <cuda_runtime.h>

#include "common.h"
#include "matrix.h"


CPUMatrix matrix_alloc_cpu(int width, int height)
{
	CPUMatrix m;
	m.width = width;
	m.height = height;
	m.elements = new float[m.width * m.height];
	return m;
}
void matrix_free_cpu(CPUMatrix &m)
{
	delete[] m.elements;
}

GPUMatrix matrix_alloc_gpu(int width, int height)
{
	GPUMatrix m;
	m.width = width;
	m.height = height;

	cudaMallocPitch((void**)&m.elements, &m.pitch, width * sizeof(float), m.height);
	
	return m;
}
void matrix_free_gpu(GPUMatrix &m) {
	cudaFree(m.elements);
}

void matrix_upload(const CPUMatrix &src, GPUMatrix &dst)
{
	int size = src.height * src.width;
	cudaMemcpy(dst.elements, src.elements, size * sizeof(float), cudaMemcpyHostToDevice);
}

void matrix_download(const GPUMatrix &src, CPUMatrix &dst)
{
	int size = src.height * src.width;
	cudaMemcpy(dst.elements, src.elements, size * sizeof(float), cudaMemcpyDeviceToHost);
}

void matrix_compare_cpu(const CPUMatrix &a, const CPUMatrix &b)
{
	int j = 0;
	for (int i = 0; i < a.height * a.width; i++) {
		if (a.elements[i] == b.elements[i]) {
			j++;
		}
	}
	if (j == (a.height * a.width)) {
		std::cout << "CPU Matrix and GPU Matrix are equal" << std::endl;
	} else {
		std::cout << "Matrices are not equal" << std::endl;
	}
}
