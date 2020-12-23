#include "conv_gpu.h"
#include "stdio.h"
#include <cuda_runtime.h>

/*  ==========================     TASK 2.2 GLOBAL MEMORY     ============================*/
__global__ void ConvKernelGlobal(const image_gpu &m, const image_gpu &n, const filterkernel_gpu &kernel) {
    /*int threadx = threadIdx.x;
    int thready = threadIdx.y;
    int blockx = blockIdx.x;
    int blocky = blockIdx.y;

    int tx = blockx * BLOCK_SIZE + threadx;
    int ty = blocky * BLOCK_SIZE + thready;

*/
}


void conv_h_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {
    for (int i = 0; i < kernel.ks; i++){ 
        printf("%f,  ", kernel.data);
    }
}

void conv_v_gpu_gmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}




/*  ==========================     TASK 2.3 SHARED MEMORY     ============================*/
void conv_h_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}

void conv_v_gpu_smem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}


/*  ==========================     TASK 2.4 CONSTANT MEMORY     ============================*/
void conv_h_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}

void conv_v_gpu_cmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}



void conv_h_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}

void conv_v_gpu_tmem(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}

void conv_h_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}

void conv_v_gpu_all(image_gpu &dst, const image_gpu &src, const filterkernel_gpu &kernel) {

}