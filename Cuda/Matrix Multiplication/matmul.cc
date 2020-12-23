#include <iostream>

#include <cuda_runtime.h>

#include "matmul.h"
#include "test.h"
#include "common.h"
#include "mul_cpu.h"
#include "mul_gpu.h"
#include "timer.h"


void print_matrix(CPUMatrix &matrix) {
	int j = 0;
	for (int i = 0; i < matrix.height * matrix.width; ++i) {
		std::cout << matrix.elements[i] << ' ';
		j++;
		if (j == matrix.width) {
			std::cout << std::endl;
			j = 0;
		}
	}
}

CPUMatrix invert2DArray(CPUMatrix m) {
	CPUMatrix n;
	return n;
}

void print_cuda_devices() {
	int nDevice;
	cudaGetDeviceCount(&nDevice);
	std::cout << nDevice << std::endl;
	for (int i = 0; i < nDevice; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		std::cout << "Compute capability (Major/Minor): \t" << prop.major << ", " << prop.minor << std::endl;
		std::cout << "Multiprocessor count: \t\t\t" << prop.multiProcessorCount << std::endl;	//return in int
		std::cout << "GPU clock rate: \t\t\t" << prop.clockRate << " kHz (" << prop.clockRate / 1000000.0 << "GHz)" << std::endl;
		std::cout << "Total global memory: \t\t\t" << prop.totalGlobalMem << " bytes ("<< prop.totalGlobalMem / 1024.0  << " MiB)"<< std::endl;	//return in bytes
		std::cout << "L2 chache size: \t\t\t" << prop.l2CacheSize << " bytes ("<< prop.l2CacheSize / 1048576.0 << " KiB)" << std::endl;			//return in bytes
		std::cout << "\n" << std::endl;
	}
}

void matmul()
{
	// === Task 3 ===
	// TODO: Allocate CPU matrices (see matrix.cc)
	//       Matrix sizes:
	//       Input matrices:
	//       Matrix M: pmpp::M_WIDTH, pmpp::M_HEIGHT
	//       Matrix N: pmpp::N_WIDTH, pmpp::N_HEIGHT
	//       Output matrices:
	//       Matrix P: pmpp::P_WIDTH, pmpp::P_HEIGHT

	CPUMatrix inMatrixCPU1 = matrix_alloc_cpu(pmpp::M_HEIGHT, pmpp::M_WIDTH);
	CPUMatrix inMatrixCPU2 = matrix_alloc_cpu(pmpp::N_WIDTH, pmpp::N_WIDTH);

	CPUMatrix outMatrixCPU = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

	// TODO: Fill the CPU input matrices with the provided test values (pmpp::fill(CPUMatrix &m, CPUMatrix &n))
	pmpp::fill(inMatrixCPU1, inMatrixCPU2);

	// TODO (Task 5): Start CPU timing here!
	timer_tp start_cpu = timer_now();

	// TODO: Run your implementation on the CPU (see mul_cpu.cc)
	matrix_mul_cpu(inMatrixCPU1, inMatrixCPU2, outMatrixCPU);

	// TODO (Task 5): Stop CPU timing here!
	timer_tp stop_cpu = timer_now();
	float cpu_comp_time = timer_elapsed(start_cpu, stop_cpu);

	// TODO: Check your matrix for correctness (pmpp::test_cpu(const CPUMatrix &p))
	pmpp::test_cpu(outMatrixCPU);


	std::cout << "\n" << std::endl;

	// === Task 4 ===
	// TODO: Set CUDA device
	int nDevice = 0;
	cudaGetDeviceCount(&nDevice);
	int userDeviceInput = nDevice - 1;

	if (userDeviceInput < nDevice) {
		cudaSetDevice(userDeviceInput);
	} else {
		printf("error: invalid device choosen\n");
	}

	// TODO: Allocate GPU matrices (see matrix.cc)
	GPUMatrix inMatrixGPU1 = matrix_alloc_gpu(pmpp::M_HEIGHT, pmpp::M_WIDTH);
	GPUMatrix inMatrixGPU2 = matrix_alloc_gpu(pmpp::N_WIDTH, pmpp::N_WIDTH);
	
	GPUMatrix outMatrixGPU = matrix_alloc_gpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);

	// TODO: Upload the CPU input matrices to the GPU (see matrix.cc)
	matrix_upload(inMatrixCPU1, inMatrixGPU1);
	matrix_upload(inMatrixCPU2, inMatrixGPU2);
	
	// TODO (Task 5): Start GPU timing here!
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
	
	// TODO: Run your implementation on the GPU (see mul_gpu.cu)
	matrix_mul_gpu(inMatrixGPU1, inMatrixGPU2, outMatrixGPU);
	
	// TODO (Task 5): Stop GPU timing here!
	cudaEventRecord(evStop, 0);
	cudaEventSynchronize(evStop);
	float elapsedTime_ms;
	
	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
	printf("CUDA processing took: %f ms\n", elapsedTime_ms);
	cudaEventDestroy(evStart);
	cudaEventDestroy(evStop);
	
	// TODO: Download the GPU output matrix to the CPU (see matrix.cc)
	CPUMatrix outputMatrixGPU = matrix_alloc_cpu(pmpp::P_WIDTH, pmpp::P_HEIGHT);
	matrix_download(outMatrixGPU, outputMatrixGPU);

	// TODO: Check your downloaded matrix for correctness (pmpp::test_gpu(const CPUMatrix &p))
	pmpp::test_gpu(outputMatrixGPU);
	
	// TODO: Compare CPU result with GPU result (see matrix.cc)
	matrix_compare_cpu(outMatrixCPU, outputMatrixGPU);
	
	//print compution time from cpu and gpu
	print_time(cpu_comp_time, elapsedTime_ms);



	// TODO (Task3/4/5): Cleanup ALL matrices and and events
	//Free CPU memory
	matrix_free_cpu(inMatrixCPU1);
	matrix_free_cpu(inMatrixCPU2);
	matrix_free_cpu(outMatrixCPU);

	//Free GPU memory
	matrix_free_gpu(inMatrixGPU1);
	matrix_free_gpu(inMatrixGPU2);
	matrix_free_gpu(outMatrixGPU);
	matrix_free_cpu(outputMatrixGPU);
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 4) 6. Where do the differences come from?
 * 
 * Answer: in PDF-File
 * 
 * 
 ************************************************************/