#include "lsqrCUDAcuSparseKernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include <cusparse_v2.h>

#define BLOCK_SIZE 32            //max threads in a block

__global__ void sqaure_vector(const double *vector, double *tmp, const int size);
__global__ void norm2(const double *in_data, double *result, int size);
__global__ void add_subtract_vector(double *a, double *b, bool operation, int size);  
__global__ void scalar_vector(double *in_data, const double scalar, const int size);
__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result);
//__global__ void matrix_vector_multiplication_sh(const int n_row, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result);


inline unsigned int div_up(unsigned int numerator, unsigned int denominator) { //numerator = zähler, denumerator = nenner
	unsigned int result = numerator / denominator;
	if (numerator % denominator) ++result;
    return result;
    
}


GPUMatrix transpose_matrix(GPUMatrix A) {
    GPUMatrix A_transpose = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    kernelCheck(__LINE__);

    size_t tempInt;
    double *buffer;

    cusparseCsr2cscEx2_bufferSize(handle, A.height, A.width, A.elementSize,
                                  A.elements, A.csrRow, A.csrCol,
                                  A_transpose.elements, A_transpose.csrCol,A_transpose.csrRow, 
                                  CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &tempInt);
    kernelCheck(__LINE__);
    cudaMalloc(&buffer, tempInt);
    kernelCheck(__LINE__);
    cusparseCsr2cscEx2(handle, A.height, A.width, A.elementSize,
                       A.elements, A.csrRow, A.csrCol,
                       A_transpose.elements, A_transpose.csrRow, A_transpose.csrCol, 
                       CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    kernelCheck(__LINE__);
    return A_transpose;
}



/*
<<<<<<<<<<-------------------- NORM ----------------------------->>>>>>>>>>>>>>
*/

__global__ void sqaure_vector(const double *vector, double *tmp, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= size) { 
        tmp[i] = 0;
    } else {
        tmp[i] = vector[i] * vector[i];
    }

    __syncthreads();
}


__global__ void norm2(const double *in_data, double *result,int size) {
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < size){
        sdata[tid] = in_data[i];        //load global data in sh_memory
    }else{
        sdata[tid] = 0; 
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    //thread 0 writes in result back to global memory
    if (tid == 0) {
        result[blockIdx.x] = sdata[0]; 
    }
}


double getNorm2(const GPUMatrix denseVector) {
    GPUMatrix tmp = matrix_alloc_gpu(denseVector.height, denseVector.width);

    int grids = div_up(denseVector.height, BLOCK_SIZE * BLOCK_SIZE);

    double *result;
    cudaMalloc(&result, grids * sizeof(double));
    
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);
    
    sqaure_vector<<<grids, dimBlock>>>(denseVector.elements, tmp.elements, tmp.height * tmp.width); 
    norm2<<<grids, dimBlock, sh_memory_size>>>(tmp.elements, result, tmp.height * tmp.width);

    
    double *values = new double[grids]; 
    cudaMemcpy(values, result, grids * sizeof(double), cudaMemcpyDeviceToHost);

    double norm = 0.0;
    for (int i= 0; i< grids; i++) {
        norm += values[i];
    }

    matrix_free_gpu(tmp);
    delete[] values;
    cudaFree(result);

    return sqrt(norm);
}


/*
<<<<<<<<<<-------------------- END NORM ----------------------------->>>>>>>>>>>>>>>>>
*/





/*
<<<<<<<<<<-------------------- ADDITION AND SUBSTRACTION ----------------------------->>>>>>>>>>>>>>>>>
*/

void get_add_subtract_vector(GPUMatrix denseA, GPUMatrix denseB, bool operation) {
    int grids = div_up(denseA.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);

    add_subtract_vector<<<grids, dimBlock>>>(denseA.elements, denseB.elements, operation, denseA.width * denseA.height);
}



__global__ void add_subtract_vector(double *a, double *b, bool operation, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    //check if index out of range of vector
    if(i >= size) return;

    if(operation == true) {
        a[i] = a[i] + b[i];

    } else {
        a[i] = a[i] - b[i];
    }
    __syncthreads();
}


/*
<<<<<<<<<<-------------------- END ADDITON AND SUBSTRACTION ----------------------------->>>>>>>>>>>>>>>>>
*/




/*
<<<<<<<<<<-------------------- MULTIPLY SCALAR ----------------------------->>>>>>>>>>>>>>>>>
*/

void multiply_scalar_vector(GPUMatrix vector, const double scalar) {
    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);

    scalar_vector<<<grids, dimBlock>>>(vector.elements, scalar, vector.height * vector.width);
}


__global__ void scalar_vector(double *in_data, const double scalar, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        in_data[i] = scalar * in_data[i];
    }
    __syncthreads();
}
/*
<<<<<<<<<<-------------------- END MULTIPLICATION SCALAR ----------------------------->>>>>>>>>>>>>>>>>
*/




/*
<<<<<<<<<<-------------------- CSR MATRIX MULTIPLY WITH DENSE VECTOR ----------------------------->>>>>>>>>>>>>>>>>
*/

__global__ void matrix_vector_multiplication(const int n_rows, const double *elements, 
                                            const int *rowPtr, const int *colIdx, const double *x, double *result) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n_rows) {
        const int row_start = rowPtr[row];
        const int row_end = rowPtr[row + 1];

        double sum = 0.0;
        for (int idx = row_start; idx < row_end; idx++) {
            int col = colIdx[idx];
            sum += elements[idx] * x[col];
        }
        result[row] = sum;
    }
    __syncthreads();
}

// __global__ void matrix_vector_multiplication_sh(const int n_row, const double *elements, const int *rowPtr, const int *colIdx, const double *x, double *result) {
    
// }


// GPUMatrix get_csr_matrix_vector_multiplication_sh(const GPUMatrix matrix, const GPUMatrix vector, GPUMatrix result) {
//     int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
//     dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
//     int sh_memory_size = BLOCK_SIZE * BLOCK_SIZE * sizeof(double);

//     matrix_vector_multiplication_sh<<<grids, dimBlock, sh_memory_size>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);
// }



void get_csr_matrix_vector_multiplication(const GPUMatrix matrix, const GPUMatrix vector, GPUMatrix result) {
    int grids = div_up(vector.height, BLOCK_SIZE * BLOCK_SIZE);
    kernelCheck(__LINE__);
    dim3 dimBlock(BLOCK_SIZE * BLOCK_SIZE);
    kernelCheck(__LINE__);
    matrix_vector_multiplication<<<grids, dimBlock>>>(matrix.height, matrix.elements, matrix.csrRow, matrix.csrCol, vector.elements, result.elements);
    kernelCheck(__LINE__);
}


/*
<<<<<<<<<<-------------------- END MATRIX VECTOR MULTIPLICATION ----------------------------->>>>>>>>>>>>>>>>>
*/

 
	
GPUMatrix lsqr_algrithm(const GPUMatrix &A, const GPUMatrix &b, const int max_iters, const double ebs) {
    //printf("-----------------------------INIZIALZATION--------------------------\n");
    int i = 0;
    double curr_err;
    double prev_err = DBL_MAX;
    kernelCheck(__LINE__);
    GPUMatrix x = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);
    GPUMatrix u = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);
    GPUMatrix v = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);
    GPUMatrix w = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);
    GPUMatrix A_transpose = transpose_matrix(A);
    kernelCheck(__LINE__);

    cudaEvent_t evStart, evStop;
	cudaEventCreate(&evStart);
 	cudaEventCreate(&evStop);
	cudaEventRecord(evStart, 0);
    

    //<<<<<< -------------- INIZIALIZATION PART -------------------->>>>>>>>>>>>>
    //beta = norm(b);
    double beta = getNorm2(b);
    kernelCheck(__LINE__);
    //printf("beta: %lf\n", beta);

    //u = b/beta;
    cudaMemcpy(u.elements, b.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
    kernelCheck(__LINE__);
    multiply_scalar_vector(u, 1 / beta);
    kernelCheck(__LINE__);

    //v = A' * u;
    get_csr_matrix_vector_multiplication(A_transpose, u, v);
    kernelCheck(__LINE__);

    //alpha = norm(v);
    double alpha = getNorm2(v);
    //printf("alpha: %lf\n", alpha);

    //v = v/alpha;
    multiply_scalar_vector(v, 1 / alpha);
    kernelCheck(__LINE__);


    //phi_hat = beta;
    double phi_hat = beta;
    //printf("phi_hat: %lf\n", phi_hat);

    //rho_hat = alpha;
    double rho_hat = alpha;
    //printf("rho_hat: %lf\n", rho_hat);

    //w = v
    cudaMemcpy(w.elements, v.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
    kernelCheck(__LINE__);
    //printf("\n------------------------- GETTING IN THE LOOP-----------------------\n");

    GPUMatrix tmp = matrix_alloc_gpu(b.height, b.width);
    for (; i < max_iters; i++) {
        //3a)
        //u = A * v - alpha * u;
        get_csr_matrix_vector_multiplication(A, v, tmp);
        kernelCheck(__LINE__);
        

        multiply_scalar_vector(u, alpha);
        kernelCheck(__LINE__);
        get_add_subtract_vector(tmp, u, false);
        kernelCheck(__LINE__);

        //beta = norm(u);
        beta = getNorm2(tmp);
        kernelCheck(__LINE__);
        //printf("beta: %lf\n", beta);
        
        
        //u = u / beta;
        cudaMemcpy(u.elements, tmp.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        multiply_scalar_vector(u, 1/beta);
        kernelCheck(__LINE__);

        //3b)
        // v = A' * u - beta * v;
        get_csr_matrix_vector_multiplication(A_transpose, u, tmp);
        kernelCheck(__LINE__);
        multiply_scalar_vector(v, beta);
        kernelCheck(__LINE__);
        get_add_subtract_vector(tmp, v, false);
        kernelCheck(__LINE__);


        //alpha = norm(v);
        alpha = getNorm2(tmp);
        kernelCheck(__LINE__);
        //printf("alpha: %lf\n", alpha);


        //v = v / alpha;
        cudaMemcpy(v.elements, tmp.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        multiply_scalar_vector(v, 1/alpha);
        kernelCheck(__LINE__);


        //rho = sqrt(rho_hat^2 + beta^2);
        double rho = sqrt(rho_hat * rho_hat + beta * beta);
        //c = rho_hat / rho;
        double c = rho_hat / rho;
        // s = beta / rho;
        double s = beta / rho;
        //theta = s * alpha;
        double theta = s * alpha;
        //rho_hat = -c * alpha;
        rho_hat = -c * alpha;
        //phi = c * phi_hat;
        double phi = c * phi_hat;
        //phi_hat = s * phi_hat;
        phi_hat = s * phi_hat;
        //printf("rho: %lf, c: %lf, s: %lf, theta: %lf, rho_hat: %lf, phi: %lf, phi_hat: %lf\n", rho, c, s, theta, rho_hat, phi, phi_hat);

        //5
        //x = x + (phi / rho) * w;
        cudaMemcpy(tmp.elements, w.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        multiply_scalar_vector(tmp, phi / rho);
        kernelCheck(__LINE__);
        get_add_subtract_vector(x, tmp, true);
        kernelCheck(__LINE__);
        

        // w = v - (theta / rho) * w;
        multiply_scalar_vector(w, (theta / rho));
        kernelCheck(__LINE__);
        cudaMemcpy(tmp.elements, v.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        get_add_subtract_vector(v, w, false);
        kernelCheck(__LINE__);


        cudaMemcpy(w.elements, v.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        cudaMemcpy(v.elements, tmp.elements, b.height * sizeof(double), cudaMemcpyDeviceToDevice);
        kernelCheck(__LINE__);
        
        get_csr_matrix_vector_multiplication(A, x, tmp);
        kernelCheck(__LINE__);
        get_add_subtract_vector(tmp, b, false);
        kernelCheck(__LINE__);
        curr_err = getNorm2(tmp);
        kernelCheck(__LINE__);
        //printf("curr_err: %lf\n", curr_err);

        double improvment = prev_err - curr_err;
        if(i%200==0)printf("line: %d size of error: %.6f \n",i,curr_err);
        if (curr_err < ebs) break;
        //printf("i: %d ------------------------NEXT ITERATION -----------------------\n", i); 
    }

    printf("LSQR using kernels finished.\n Iterations num: %d\n Size of error: %.6f\n",i,curr_err);
    cudaEventRecord(evStop, 0);
    cudaEventSynchronize(evStop);
    float elapsedTime_ms;
    cudaEventElapsedTime(&elapsedTime_ms, evStart, evStop);
    printf("LSQR using our kernels took: %f ms\n", elapsedTime_ms);
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);

    matrix_free_gpu(tmp);
    matrix_free_gpu(u);
    matrix_free_gpu(v);
    matrix_free_gpu(w);
    
    //printf("----------------------END KERNEL------------------------\n\n\n");
    return x; 
}


CPUMatrix sparseLSQR_with_kernels(const CPUMatrix &A, const CPUMatrix &b, const int max_iters, const double ebs) {
    kernelCheck(__LINE__);
    CPUMatrix resultCPU = matrix_alloc_cpu(b.height, b.width);
    kernelCheck(__LINE__);
    GPUMatrix resultGPU = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);

    GPUMatrix A_gpu = matrix_alloc_sparse_gpu(A.height, A.width, A.elementSize, A.rowSize, A.columnSize);
    kernelCheck(__LINE__);
    GPUMatrix b_gpu = matrix_alloc_gpu(b.height, b.width);
    kernelCheck(__LINE__);

    /* upload Matrix, vector */
    matrix_upload_cuSparse(A, A_gpu);
    kernelCheck(__LINE__);
    matrix_upload(b, b_gpu);
    kernelCheck(__LINE__);
    resultGPU = lsqr_algrithm(A_gpu, b_gpu, max_iters, ebs);
    /* Download result */
    matrix_download(resultGPU, resultCPU);
    kernelCheck(__LINE__);
    
    /* free GPU memory */
    matrix_free_sparse_gpu(A_gpu);
    matrix_free_gpu(b_gpu);
    matrix_free_gpu(resultGPU);

    return resultCPU;
}


void printVectorKernel(int iteration,GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height, x.width);
	matrix_download(x ,tempCPUMatrix);
	//printf("iteration number: %d\n", iteration);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}

void printValuesKernel(GPUMatrix x, const char *name) {
    printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_sparse_cpu(x.height, x.width, x.elementSize, x.rowSize, x.columnSize);
    matrix_download_cuSparse(x ,tempCPUMatrix);
    

    for(int i = 0; i < x.elementSize; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
    }
    printf("\n Row:");

    for(int i = 0; i < x.rowSize; i++){
		printf("%d ", tempCPUMatrix.csrRow[i]);
    }
    printf("\n Col:");
    for(int i = 0; i < x.columnSize; i++){
		printf("%d ", tempCPUMatrix.csrCol[i]);
    }
    printf("\n");
}


void kernelCheck(int line){
	const cudaError_t err = cudaGetLastError();                            
	if (err != cudaSuccess) {                                              
    	const char *const err_str = cudaGetErrorString(err);               
    	std::cerr << "Cuda error in " << __FILE__ << ":" << line - 1   
            << ": " << err_str << " (" << err << ")" << std::endl;   
            exit(EXIT_FAILURE);                                                                    
	}
}