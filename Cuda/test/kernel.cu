
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include "ppm_lib.h"
#include "Constant.h"

using namespace std;

#define BLOCK_WIDTH 16 
#define BLOCK_HEIGHT 16
#define ITERATION 1

#define PATH "C:/Users/jeanf/Desktop/gare_parallelisme2.ppm"
#define PATH_OUT_QESTION_A "C:/Users/jeanf/Desktop/mon_imageA.ppm"
#define PATH_OUT_QESTION_D "C:/Users/jeanf/Desktop/mon_imageD.ppm"
#define PATH_OUT_QESTION_C "C:/Users/jeanf/Desktop/mon_imageC.ppm"
#define PATH_OUT_SOBEL "C:/Users/jeanf/Desktop/mon_imageSobel.ppm"

cudaError_t filterWithCuda(PPMImage *in, PPMImage *out, int *filter, int div, bool isOpti, bool isSobel);
void applyFilter(PPMImage *in, PPMImage *out, int filter[], int div);


//---------------------------------
// Kernel with optimisation
//---------------------------------
__global__ void filterKernelOpti(const PPMPixel *globalIn, PPMPixel *out, const int globalWidth, const int globalHeight, const int *globalFilter, const int globalDiv)
{

	unsigned int localX = threadIdx.x;
	unsigned int localY = threadIdx.y;

	unsigned int x = blockIdx.x*blockDim.x + localX;
	unsigned int y = blockIdx.y*blockDim.y + localY;

	int filter[25];
	int i, j;
	
	// Copy into the register
	unsigned const int width = globalWidth;
	unsigned const int height = globalHeight;
	unsigned const int div = globalDiv;

	// Copy the filter into local memory
	for (i = 0; i < 5; i++) {
		for(j =0; j<5; j++) {
			filter[j * 5 + i] = globalFilter[j * 5 + i];
		}
	}

	int x2, y2;

	// Copy a part of the image into shared memory
	__shared__ PPMPixel in[BLOCK_HEIGHT+4][BLOCK_WIDTH+4];
	if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
		for (y2 = -2; y2 <= 2; y2++) {
			for (x2 = -2; x2 <= 2; x2++) {
				in[localY+y2+2][localX+x2+2] = globalIn[(y + y2)*width + (x + x2)];
			}
		}
	}
	else if ((x == 0 || x == 1 || x == width - 2 || x == width - 1) && y < height) {
		in[localY][localX] = globalIn[y*width + x];
	}
	else if ((y == 0 || y == 1 || y == height - 2 || y == height - 1) && x < width) {
		in[localY][localX] = globalIn[y*width + x];
	}

	__syncthreads();


	for (i = 0; i < ITERATION; i++) {
		if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
			int gridCounter = 0;
			int finalR = 0;
			int finalG = 0;
			int finalB = 0;

			for (y2 = -2; y2 <= 2; y2++) {
				for (x2 = -2; x2 <= 2; x2++) {
					finalR += in[localY + y2 + 2][localX + x2 + 2].red * filter[gridCounter];
					finalG += in[localY + y2 + 2][localX + x2 + 2].green * filter[gridCounter];
					finalB += in[localY + y2 + 2][localX + x2 + 2].blue * filter[gridCounter];
					gridCounter++;
				}
			}
			finalR /= div;
			finalG /= div;
			finalB /= div;
			out[y*width + x].red = finalR;
			out[y*width + x].green = finalG;
			out[y*width + x].blue = finalB;
		}
		else if ((x == 0 || x == 1 || x == width - 2 || x == width - 1) && y < height) {
			out[y*width + x] = in[localY][localX];
		}
		else if ((y == 0 || y == 1 || y == height - 2 || y == height - 1) && x < width) {
			out[y*width + x] = in[localY][localX];
		}
	}
}


//---------------------------------
// Simple Kernel
//---------------------------------
__global__ void filterKernel(PPMPixel *in, PPMPixel *out, int width, int height, int *filter, int div)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i;

	int x2, y2;
	int gridCounter = 0;
	int finalR = 0;
	int finalG = 0;
	int finalB = 0;

	for (i = 0; i < ITERATION; i++) {
		if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
			gridCounter = 0;
			finalR = 0;
			finalG = 0;
			finalB = 0;
			for (y2 = -2; y2 <= 2; y2++) {
				for (x2 = -2; x2 <= 2; x2++) {
					finalR += in[(y + y2)*width + x + x2].red * filter[gridCounter];
					finalG += in[(y + y2)*width + x + x2].green * filter[gridCounter];
					finalB += in[(y + y2)*width + x + x2].blue * filter[gridCounter];
					gridCounter++;
				}
			}
			finalR /= div;
			finalG /= div;
			finalB /= div;
			out[y*width + x].red = finalR;
			out[y*width + x].green = finalG;
			out[y*width + x].blue = finalB;
		}
		else if ((x == 0 || x == 1 || x == width - 2 || x == width - 1) && y < height) {
			out[y*width + x] = in[y*width + x];
		}
		else if ((y == 0 || y == 1 || y == height - 2 || y == height - 1) && x < width) {
			out[y*width + x] = in[y*width + x];
		}
	}
}


//---------------------------------
// Simple Kernel for Sobel
//---------------------------------
__global__ void filterKernelForSobel(PPMPixel *in, PPMPixel *out, int width, int height, int *filter, int div)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int i;

	int x2, y2;
	int gridCounter = 0;
	int finale = 0;

	if (x >= 2 && x < width - 2 && y >= 2 && y < height - 2) {
		gridCounter = 0;
		finale = 0;

		for (i = 0; i < ITERATION; i++) {
			for (y2 = -2; y2 <= 2; y2++) {
				for (x2 = -2; x2 <= 2; x2++) {
					int grayColor = (in[(y + y2)*width + x + x2].red + in[(y + y2)*width + x + x2].green + in[(y + y2)*width + x + x2].blue) / 3;
					finale += grayColor * filter[gridCounter];
					gridCounter++;
				}
			}
			finale /= div;
			if (finale < 0) {
				out[y*width + x].red = 0;
				out[y*width + x].green = 0;
				out[y*width + x].blue = 0;
			}
			else if (finale > 255) {
				out[y*width + x].red = 255;
				out[y*width + x].green = 255;
				out[y*width + x].blue = 255;
			}
			else {
				out[y*width + x].red = finale;
				out[y*width + x].green = finale;
				out[y*width + x].blue = finale;
			}
		}
	}
}


//---------------------------------
// Main
//---------------------------------
int main()
{
	PPMImage *imageIn;
	imageIn = readPPM(PATH);

// Question A
	PPMImage *imageOut;
	imageOut = readPPM(PATH);
	applyFilter(imageIn, imageOut, Soften, SoftenDiv);
	writePPM(PATH_OUT_QESTION_A, imageOut);

// Question C CUDA
	PPMImage *imageOutCudaC = readPPM(PATH);
	cudaError_t cudaStatusC = filterWithCuda(imageIn, imageOutCudaC, Soften, SoftenDiv, false, false);
    if (cudaStatusC != cudaSuccess) {
        fprintf(stderr, "filterWithCuda failed!\n");
        return 1;
    }
	writePPM(PATH_OUT_QESTION_C, imageOutCudaC);
	cudaStatusC = cudaDeviceReset();
	if (cudaStatusC != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

// Question D CUDA
	PPMImage *imageOutCudaD = readPPM(PATH);
	cudaError_t cudaStatusD = filterWithCuda(imageIn, imageOutCudaD, Soften, SoftenDiv, true, false);
	if (cudaStatusD != cudaSuccess) {
		fprintf(stderr, "filterWithCuda failed!\n");
		return 1;
	}
	writePPM(PATH_OUT_QESTION_D, imageOutCudaD);
	cudaStatusD = cudaDeviceReset();
    if (cudaStatusD != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!\n");
        return 1;
    }


	// CUDA pour Sobel
	PPMImage *imageOutCudaSobel = readPPM(PATH);
	cudaError_t cudaStatusSobel = filterWithCuda(imageIn, imageOutCudaSobel, VerticalSobel, HorizontalSobelDiv, false, true);
	if (cudaStatusSobel != cudaSuccess) {
		fprintf(stderr, "filterWithCuda failed!\n");
		return 1;
	}
	writePPM(PATH_OUT_SOBEL, imageOutCudaSobel);
	cudaStatusSobel = cudaDeviceReset();
	if (cudaStatusSobel != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}
    return 0;
}


cudaError_t filterWithCuda(PPMImage *in, PPMImage *out, int *filter, int div, bool isOpti, bool isSobel) {
	int size = in->x * in->y;
	int *dev_filter;
	PPMPixel *dev_in;
	PPMPixel *dev_out;
	cudaError_t cudaStatus;

	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}


// Cuda malloc
	// PPMPixel in  .
	cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(PPMPixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed 0 !\n");
		goto Error;
	}

	// PPMPixel out
	cudaStatus = cudaMalloc((void**)&dev_out, size * sizeof(PPMPixel));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed 0 !\n");
		goto Error;
	}

	// Filter
	cudaStatus = cudaMalloc((void**)&dev_filter, 25 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed 0 !\n");
		goto Error;
	}

// Cuda copy
	// Copy PPMPixel in
	cudaStatus = cudaMemcpy(dev_in, in->data, size * sizeof(PPMPixel), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed 3 !\n");
		goto Error;
	}
	
	// Copy Filter
	cudaStatus = cudaMemcpy(dev_filter, filter, 25 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed 3 !\n");
		goto Error;
	}

	
// Launch a kernel
	// blockDim
	unsigned int blockDimX = BLOCK_WIDTH;              // X ~ COL
	unsigned int blockDimY = BLOCK_HEIGHT;              // Y ~ ROW


	printf("\n-------------------------------------\n");
	if (isSobel == false) {
		if (isOpti == false) {
			printf("Filtre non optimise \n");
		}
		else {
			printf("Filtre optimise \n");
		}
	}
	else {
		printf("Filtre de Sobel non optimise \n");
	}
	printf("\n-------------------------------------\n");

	// gridDim
	printf("x = %d, y = %d\n", in->x, in->y);
	unsigned int gridDimX =	ceil((float)in->x / (float)blockDimX);   // X ~ COL 31 62
	unsigned int gridDimY = ceil((float)in->y / (float)blockDimY);   // Y ~ ROW

	dim3  GRID(gridDimX, gridDimY, 1);
	dim3  BLOCK(blockDimX, blockDimY, 1);

	printf("Grid  Dimensions: (%d, %d)\n", gridDimX, gridDimY);
	printf("Block Dimensions: (%d, %d)\n\n", blockDimX, blockDimY);
	
	// Timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Launch kernel
	cudaEventRecord(start);
	if(isOpti==false && isSobel == false)
		filterKernel << <GRID, BLOCK >> >(dev_in, dev_out, in->x, in->y, dev_filter, div);
	else if (isOpti == true && isSobel == false)
		filterKernelOpti << <GRID, BLOCK >> >(dev_in, dev_out, in->x, in->y, dev_filter, div);
	else
		filterKernelForSobel << <GRID, BLOCK >> >(dev_in, dev_out, in->x, in->y, dev_filter, div);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "filterKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	cudaEventRecord(stop);

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching filterKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(out->data, dev_out, size * sizeof(PPMPixel), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	out->x = in->x;
	out->y = in->y;

	// Timer
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Elapsed time : %f ms\n", milliseconds);

	printf("-------------------------------------\n\n\n");

Error:
	cudaFree(dev_in);
	cudaFree(dev_out);
	cudaFree(dev_filter);

	return cudaStatus;
}




void applyFilter(PPMImage *in, PPMImage *out, int filter[], int div) {
	int x, y, x2, y2;
	int width = in->x;

	for (y = 2; y<in->y - 2; y++) {
		for (x = 2; x<in->x - 2; x++) {
			int gridCounter = 0;
			int finalR = 0;
			int finalG = 0;
			int finalB = 0;
			for (y2 = -2; y2 <= 2; y2++) {
				for (x2 = -2; x2 <= 2; x2++) {
					finalR += in->data[(y + y2)*width + (x + x2)].red * filter[gridCounter];
					finalG += in->data[(y + y2)*width + (x + x2)].green * filter[gridCounter];
					finalB += in->data[(y + y2)*width + (x + x2)].blue * filter[gridCounter];
					gridCounter++;
				}
			}
			finalR /= div;
			finalG /= div;
			finalB /= div;
			out->data[y*width + x].red = finalR;
			out->data[y*width + x].green = finalG;
			out->data[y*width + x].blue = finalB;
		}
	}
}