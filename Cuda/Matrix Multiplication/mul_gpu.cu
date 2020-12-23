#include <cuda_runtime.h>

// NOTE: if you include stdio.h, you can use printf inside your kernel
#include <stdio.h>
#include "device_launch_parameters.h"

#include "common.h"
#include "matrix.h"
#include "mul_gpu.h"

// TODO (Task 4): Implement matrix multiplication CUDA kernel
__global__ void multiplication_kernel(GPUMatrix m, GPUMatrix n, GPUMatrix out) {
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float pValue = 0;

	for (int k = 0; k < m.width; k++) {
		float Melement = m.elements[ty * m.width + k];
		float Nelement = n.elements[k * n.width + tx];
		
		pValue += Melement * Nelement;
	}
	int out_index = ty * out.width + tx;
	out.elements[ty * out.width + tx] = pValue;
}



void matrix_mul_gpu(const GPUMatrix &m, const GPUMatrix &n, GPUMatrix &p)
{
	// TODO (Task 4): Determine execution configuration and call CUDA kernel
	dim3 grid(1, 1);		//div_up(m.height, n.width);
	dim3 dimBlock(m.width, n.height);

	multiplication_kernel<<<grid, dimBlock >>>(m, n, p);
	
	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess)
		printf("Error: %s\n", cudaGetErrorString(err));

	cudaDeviceSynchronize();
}
