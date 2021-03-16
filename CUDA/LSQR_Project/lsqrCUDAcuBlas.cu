#include <iostream>
#include <stdio.h>
#include "lsqr.h"
#include "lsqrCUDAcuBlas.h"
#include "matrix.h"
#include <cusparse.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h> //cuda basic linear algebra subroutine library
#define BLOCK_SIZE 32

/*
	A normal LSQR implemtation - GPU Matrix A, Vector b (as 1*n matrix)
	u,v,w,x are matrix that is allocated before hand to use for cuBLAS computations 
	vector x needs to be initialzed to 0.
*/
CPUMatrix cublasLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs, int max_iterations){
	cublasHandle_t handle;
	cublasCreate(&handle);
    GPUMatrix tempGpuMatrixA = matrix_alloc_gpu(A.height,A.width);
	matrix_upload(A,tempGpuMatrixA);
	GPUMatrix gpuMatrixA = matrix_alloc_gpu(A.height,A.width);
	matrix_upload(A,gpuMatrixA);

	double tempDouble = 1.0;
	double tempDouble2 = 0.0;
	cublasDgeam(handle,CUBLAS_OP_T,CUBLAS_OP_N,A.height,A.width,&tempDouble,tempGpuMatrixA.elements,A.width,&tempDouble2,tempGpuMatrixA.elements,A.width,gpuMatrixA.elements,A.width);

	matrix_free_gpu(tempGpuMatrixA);

	GPUMatrix gpuVectorb = matrix_alloc_gpu(b.height,b.width);
    matrix_upload(b,gpuVectorb);
    GPUMatrix u = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix v = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix w = matrix_alloc_gpu(b.height,b.width);
    GPUMatrix x = matrix_alloc_gpu(b.height,b.width);
	GPUMatrix tempVector = matrix_alloc_gpu(b.height,b.width);
	CPUMatrix fillingToX = matrix_alloc_cpu(b.height,b.width);
	for(int i=0;i<b.height;i++){
		fillingToX.elements[i]=0;
	}
	matrix_upload(fillingToX,x);
	cublasDestroy(handle);

	return cublasLSQR_aux(gpuMatrixA,gpuVectorb,u,v,w,x,tempVector,ebs,max_iterations);
}

	


CPUMatrix cublasLSQR_aux(const GPUMatrix &A, const GPUMatrix &b,GPUMatrix &u,GPUMatrix &v,GPUMatrix &w,GPUMatrix &x,GPUMatrix &tempVector,double ebs,int max_iterations){
	double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
	//init stage
	//beta = norm(b)
	cublasDnrm2(handle, b.height, b.elements,1,&beta); 
	//u = b/beta
	cudaMemcpy (u.elements, b.elements, b.height*sizeof(double), cudaMemcpyDeviceToDevice);
	tempDouble = 1/beta;
	cublasDscal(handle, u.height,&tempDouble,u.elements,1);
	//v = A'*u
	tempDouble = 0.0;
	tempDouble2 = 1.0;
	cublasDgemv (handle, CUBLAS_OP_T, A.width, A.height,&tempDouble2,A.elements, A.width, u.elements,1,&tempDouble, v.elements, 1);
	//alpha = norm(v)
	cublasDnrm2(handle, v.height, v.elements,1,&alpha); 
	//v = v/alpha;
	tempDouble = 1/alpha;
	cublasDscal(handle, v.height,&tempDouble,v.elements,1);
	//w = v;
	cudaMemcpy (w.elements, v.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
	phi_tag = beta; rho_tag = alpha;

	int i = 0;
	while(true){
		//next bidiagonlization
		// u = A * v - alpha * u;
		tempDouble = alpha*(-1.0);
		tempDouble2 = 1.0;
		cublasDgemv (handle, CUBLAS_OP_N, A.width,A.height,&tempDouble2,A.elements, A.width, v.elements,1,&tempDouble, u.elements, 1);
		//beta = norm(u);
		cublasDnrm2(handle, u.height, u.elements,1,&beta); 
		// u = u / beta;
		tempDouble = 1/beta;
		cublasDscal(handle, u.height,&tempDouble,u.elements,1);
		// v = A' * u - beta * v;
		tempDouble = (-1.0)*beta;
		tempDouble2 = 1.0;
		cublasDgemv (handle, CUBLAS_OP_T, A.width,A.height,&tempDouble2,A.elements, A.width, u.elements,1, &tempDouble, v.elements, 1);
		//alpha = norm(v)
		cublasDnrm2(handle, v.height, v.elements,1,&alpha); 
		//v = v/alpha;
		tempDouble = 1/alpha;
		cublasDscal(handle, v.height,&tempDouble,v.elements,1);
		//next orthogonal transformation
		rho = sqrt(pow (rho_tag, 2.0) + pow (beta, 2.0));
		c = rho_tag / rho;
		s = beta / rho;
		theta = s * alpha;
		rho_tag = (-1) * c * alpha;
		phi = c * phi_tag;
		phi_tag = s * phi_tag;
		//updating x,w
		//x =  (phi / rho) * w + x;             (in cublas : x is y, w is x)
		tempDouble = phi / rho;
		cublasDaxpy(handle,w.height,&tempDouble,w.elements, 1,x.elements, 1);
		//	w = v - (theta / rho) * w ;
		tempDouble = (-1.0) * (theta / rho);
		cudaMemcpy (tempVector.elements, v.elements, v.height*sizeof(double), cudaMemcpyDeviceToDevice);
		cublasDaxpy(handle,tempVector.height,&tempDouble,w.elements, 1,tempVector.elements, 1);
		cudaMemcpy (w.elements, tempVector.elements, tempVector.height*sizeof(double), cudaMemcpyDeviceToDevice);
		//check for convergence
		//residual = norm(A*x - b);
		cudaMemcpy (tempVector.elements, b.elements, tempVector.height*sizeof(double), cudaMemcpyDeviceToDevice);
		//Ax - b (result in tempVector)
		tempDouble = -1.0;
		tempDouble2 = 1.0;
		cublasDgemv (handle, CUBLAS_OP_N, A.width,A.height,&tempDouble2,A.elements, A.width, x.elements,1,&tempDouble, tempVector.elements, 1);
		cublasDnrm2(handle, tempVector.height,tempVector.elements,1,&curr_err); 
		if(i%200==0) printf("line: %d size of error: %.6f\n",i,curr_err);
		i++;
		if(i==max_iterations || curr_err<ebs) break;
	}
	printf("LSQR using cuBLAS finished.\n Iterations num: %d\n Size of error: %.6f\n",i,curr_err);
	cudaEventRecord(evStop, 0);
 	cudaEventSynchronize(evStop);
 	float elapsedTime_ms;
 	cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
 	printf("LSQR using cuBlas library took: %f ms\n", elapsedTime_ms);
 	cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
	CPUMatrix result = matrix_alloc_cpu(x.height,x.width);
	matrix_free_gpu(u);
	matrix_free_gpu(v);
	matrix_free_gpu(w);
	matrix_free_gpu(x);
	matrix_free_gpu(tempVector);
	matrix_download(x,result);
	cublasDestroy(handle);
	return result;
}



void cuBLASCheck(int line){
	const cudaError_t err = cudaGetLastError();                            
	if (err != cudaSuccess) {                                              
    	const char *const err_str = cudaGetErrorString(err);               
    	std::cerr << "Cuda error in " << __FILE__ << ":" << line - 1   
            << ": " << err_str << " (" << err << ")" << std::endl;   
            exit(EXIT_FAILURE);                                                                    
	}
}





void printVector(int iteration,GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,tempCPUMatrix);
	//printf("iteration number: %d\n", iteration);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}