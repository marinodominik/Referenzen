#include "filtering.h"
#include "image.h"
#include "common.h"
#include "conv_cpu.h"
#include "conv_gpu.h"

void filtering(const char *imgfile, int ks)
{
	// === Task 1 ===
	// TODO: Load image
	image_cpu *imageCPU_src = new image_cpu(imgfile);
	image_cpu *imageCPU_dst = new image_cpu(imageCPU_src->width, imageCPU_src->height);
	

	// TODO: Generate gaussian filter kernel
	filterkernel_cpu *kernel = new filterkernel_cpu(ks);
	for (int i = 0; i < ks; i++) {
        printf("%f, ", kernel->data[i]);
    }
	printf("\n");

	// TODO: Blur image on CPU
	conv_h_cpu(*imageCPU_dst, *imageCPU_src, *kernel);
	conv_v_cpu(*imageCPU_dst, *imageCPU_src, *kernel);

	const char *saveImg_path = "/gris/gris-f/homelv/dmarino/Exercises/ex2/out_cpu.ppm";
	imageCPU_dst->save(saveImg_path);

	// === Task 2 ===
	//TODO Generate GPU images
	image_gpu *imgGPU_src = new image_gpu(imageCPU_src->width, imageCPU_src->height);
	image_gpu *imgGPU_kernel = new image_gpu(imageCPU_src->width, imageCPU_src->height);
	filterkernel_gpu *kernelGPU = new filterkernel_gpu(ks);

	//TODO Upload CPU image to GPU
	imageCPU_src->upload(*imgGPU_src);
	kernel->upload(*kernelGPU);

	// TODO: Blur image on GPU (Global memory)
	conv_h_gpu_gmem(*imgGPU_src, *imgGPU_kernel, *kernelGPU);



	image_cpu *imgGPU = new image_cpu(imageCPU_src->width, imageCPU_src->height);
	imgGPU->download(*imgGPU_src);
	saveImg_path = "/gris/gris-f/homelv/dmarino/Exercises/ex2/out_gpu_gmem.ppm";
	imgGPU->save(saveImg_path);

	// === Task 3 ===
	// TODO: Blur image on GPU (Shared memory)

	// === Task 4 ===
	// TODO: Blur image on GPU (Constant memory)

	// === Task 5 ===
	// TODO: Blur image on GPU (L1/texture cache)

	// === Task 6 ===
	// TODO: Blur image on GPU (all memory types)
}


/************************************************************
 * 
 * TODO: Write your text answers here!
 * 
 * (Task 7) nvprof output
 * 
 * Answer: TODO
 * 
 ************************************************************/
