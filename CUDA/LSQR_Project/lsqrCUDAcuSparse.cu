#include "lsqrCUDAcuSparse.h"
void printVectorj(int iteration,GPUMatrix x, const char* name);
CPUMatrix cusparseLSQR(const CPUMatrix &A, const CPUMatrix &b, double ebs,int max_iterations){
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    GPUMatrix u,v,w,x,GPUb,tempVector;
    initGPUVectors(b,u,v,w,x,GPUb,tempVector);
    matrix_upload(b,GPUb);
    CPUMatrix res = cusparseLSQR_aux(A,GPUb,u,v,w,x,tempVector,ebs,max_iterations);
    cleanGPUVectors(u,v,w,x,GPUb,tempVector);
    return res; 
}

CPUMatrix cusparseLSQR_aux(const CPUMatrix &A, const GPUMatrix &VECb,GPUMatrix &VECu,GPUMatrix &VECv,GPUMatrix &VECw,GPUMatrix &VECx,GPUMatrix &tempVector,double ebs,int max_iterations){
    double beta, alpha, phi, phi_tag, rho, rho_tag, c, s, theta, tempDouble, tempDouble2,curr_err;
    size_t tempInt;
    double *buffer;
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    cusparseSpMatDescr_t spMatrixA;
    cusparseDnVecDescr_t u,v,x,tempDense;
    GPUMatrix GPUA =  matrix_alloc_sparse_gpu(A.height,A.width,A.elementSize,A.rowSize,A.columnSize);
    matrix_upload_cuSparse(A,GPUA);
    cusparseCreateCsr(&spMatrixA,GPUA.height,GPUA.width,GPUA.elementSize,GPUA.csrRow,GPUA.csrCol,GPUA.elements,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F);
    cusparseCreateDnVec(&u,VECb.height,VECu.elements,CUDA_R_64F);
    cusparseCreateDnVec(&v,VECb.height,VECv.elements,CUDA_R_64F);
    cusparseCreateDnVec(&x,VECb.height,VECx.elements,CUDA_R_64F);
    cusparseCreateDnVec(&tempDense,VECb.height,tempVector.elements,CUDA_R_64F);
	//init stage
    //beta = norm(b)
    beta = getNorm2(VECb);
    //u = b/beta
    cudaMemcpy (VECu.elements,VECb.elements, VECu.height*sizeof(double), cudaMemcpyDeviceToDevice);
    multiply_scalar_vector(VECu,1/beta);
    //v = A'*u
    tempDouble = 1; tempDouble2 = 0;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&tempInt);
    cudaMalloc(&buffer, tempInt);
    cusparseSpMV(handle, CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
    //alpha = norm(v)
    alpha = getNorm2(VECv);
    //v = v/alpha;
    multiply_scalar_vector(VECv,1/alpha);
    //w = v;
    cudaMemcpy (VECw.elements,VECv.elements, VECv.height*sizeof(double), cudaMemcpyDeviceToDevice);
    phi_tag = beta; rho_tag = alpha;
	int i = 0;
	while(true){
		//next bidiagonlization
        // u = A * v - alpha * u;

        tempDouble = 1; tempDouble2 = (-1)*alpha;
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,v,&tempDouble2,u,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        //beta = norm(u);
        beta = getNorm2(VECu);
        // u = u / beta;
        multiply_scalar_vector(VECu,1/beta);
        // v = A' * u - beta * v;
        tempDouble = 1; tempDouble2 = (-1)*beta;
        cusparseSpMV(handle,CUSPARSE_OPERATION_TRANSPOSE,&tempDouble,spMatrixA,u,&tempDouble2,v,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        //alpha = norm(v)
        alpha = getNorm2(VECv);
        //v = v/alpha;
        multiply_scalar_vector(VECv,1/alpha);
        //next orthogonal transformation
		rho = sqrt(pow (rho_tag, 2.0) + pow (beta, 2.0));
		c = rho_tag / rho;
		s = beta / rho;
		theta = s * alpha;
		rho_tag = (-1) * c * alpha;
		phi = c * phi_tag;
		phi_tag = s * phi_tag;
        //updating x,w
        cudaMemcpy (tempVector.elements,VECw.elements, VECw.height*sizeof(double), cudaMemcpyDeviceToDevice);
        multiply_scalar_vector(tempVector,phi/rho); 
        cuSPARSECheck(__LINE__);
        //x = x + (phi / rho) * w ;          
        get_add_subtract_vector(VECx,tempVector,true);
        cuSPARSECheck(__LINE__);
        //printDenseVector(x,"x",tempPrint);
        //	w = -(theta / rho) * w + v;
        multiply_scalar_vector(VECw,(theta/rho)*(-1)); 
        cuSPARSECheck(__LINE__);
        get_add_subtract_vector(VECw,VECv,true);
        cuSPARSECheck(__LINE__);

        //check for convergence
        tempDouble = 1; tempDouble2 = (-1);
        cudaMemcpy (tempVector.elements,VECb.elements, VECb.height*sizeof(double), cudaMemcpyDeviceToDevice);
        cuSPARSECheck(__LINE__);
        cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&tempDouble,spMatrixA,x,&tempDouble2,tempDense,CUDA_R_64F,CUSPARSE_CSRMV_ALG1,&buffer);
        cuSPARSECheck(__LINE__);
        //residual = norm(A*x - b);
        //Ax - b (result in tempDense)
        curr_err = getNorm2(tempVector);
        cuSPARSECheck(__LINE__);
    
        if(i % 200 ==0) printf("line: %d size of error: %.6f \n",i,curr_err);i++;
        if(i==max_iterations) break;
        if(curr_err < ebs) break;
    }
    printf("LSQR using cuSPARSE finished.\n Iterations num: %d\n Size of error: %.6f\n",i,curr_err);
    CPUMatrix result = matrix_alloc_cpu(VECb.height,VECb.width);
    cusparseDnVecGetValues(x,(void**)&tempVector.elements);
    matrix_download(tempVector,result);
    cusparseClean(handle,spMatrixA,u,v,x,tempDense);
	return result;
}

void cusparseClean(cusparseHandle_t handle, cusparseSpMatDescr_t &A,cusparseDnVecDescr_t u, cusparseDnVecDescr_t v,cusparseDnVecDescr_t x, cusparseDnVecDescr_t tempVector){
    cusparseDestroyDnVec(u);
    cusparseDestroyDnVec(v);
    cusparseDestroyDnVec(x);
    cusparseDestroyDnVec(tempVector);
    cusparseDestroy(handle);
    cuSPARSECheck(__LINE__);
}
void initGPUVectors(const CPUMatrix &b, GPUMatrix &u,GPUMatrix &v, GPUMatrix& x, GPUMatrix &w, GPUMatrix &GPUb, GPUMatrix &tempVector){
    u = matrix_alloc_gpu(b.height,b.width);
    v = matrix_alloc_gpu(b.height,b.width);
    w = matrix_alloc_gpu(b.height,b.width);
    x = matrix_alloc_gpu(b.height,b.width);
    GPUb = matrix_alloc_gpu(b.height,b.width);
    tempVector = matrix_alloc_gpu(b.height,b.width);
}
void cleanGPUVectors(GPUMatrix& u,GPUMatrix &v, GPUMatrix &x, GPUMatrix &w, GPUMatrix &GPUb, GPUMatrix &tempVector){
    matrix_free_gpu(u);
    matrix_free_gpu(v);
    matrix_free_gpu(w);
    matrix_free_gpu(x);
    matrix_free_gpu(GPUb);
    matrix_free_gpu(tempVector);
}


void cuSPARSECheck(int line){
    const cudaError_t err = cudaGetLastError();                            
	if (err != cudaSuccess) {                                              
    	const char *const err_str = cudaGetErrorString(err);               
    	std::cerr << "Cuda error in " << __FILE__ << ":" << line - 1   
            << ": " << err_str << " (" << err << ")" << std::endl;   
            exit(EXIT_FAILURE);                                                                    
	}
}
    
void printVector(GPUMatrix x, const char* name){
	printf("%s: ",name);
	CPUMatrix tempCPUMatrix = matrix_alloc_cpu(x.height,x.width);
	matrix_download(x,tempCPUMatrix);
	for(int i = 0; i < tempCPUMatrix.height; i++){
		printf("%lf ", tempCPUMatrix.elements[i]);
	}
	printf("\n");
}