/**
 * particlesort.cu
 * GP-GPU CUDA implementation of ParticleSort
 * implementation by Michael Barger (bargerm@cs.pdx.edu)
 * algorithm by Max Orhai
 * under mentorship of Professor Black, Portland State U
 * November, 2012
 */


#ifndef PARTICLESORT_CU
#define PARTICLESORT_CU

#include <cuda.h>
#include <stdio.h>
#include "../testharness/testharness.h"

#define BLOCK 416
#define BUFFER_SIZE 16
#define MAX_MOMENTUM 0xF
#define MOMENTUM_INIT 0xF0000000
#define MOMENTUM_WIDTH 4
#define COLOR_WIDTH 32 - MOMENTUM_WIDTH
#define COLOR_MASK 0x0fffffff
#define TRUE 1
#define BOOST 1
#define ENTROPY 1
#define FALSE 0
#define INCREASE_MOMENTUM(p) ((p).momentum=min((p).momentum+BOOST,MAX_MOMENTUM))
#define INCREASE_MOMENTUM_PTR(p) ((p)->momentum=min((p)->momentum+BOOST,MAX_MOMENTUM))
#define DECREASE_MOMENTUM(p) ((p).momentum=max((p).momentum-ENTROPY,0))
#define DECREASE_MOMENTUM_PTR(p) ((p)->momentum=max((p)->momentum-ENTROPY,0))
#define RESET(p) (p).color=0;(p).momentum=0


/// CUDA DEVICE KERNEL ////////////////////////////////////////////////////////////////////
extern "C" __global__ void ParticleSort (unsigned int *g, unsigned int *b, unsigned int *f, unsigned long size)
{
}

/// CUDA HOST /////////////////////////////////////////////////////////////////////////////
static void ErrorCheck (cudaError_t cerr, const char *str);
__device__ unsigned int *global_mem;
__device__ unsigned int *transblock_buffers;
__device__ unsigned int *buffer_flags;

extern "C" void sort (unsigned int *buffer, unsigned long size)
{
	dim3 grid (1);
	dim3 block (BLOCK);
	size_t global_mem_size = size * sizeof(int);
	size_t transblock_buffers_size = BUFFER_SIZE * (block.x - 1) * sizeof(int);
	size_t buffer_flags_size = (block.x - 1) * sizeof(int) * 2;

	ErrorCheck(cudaMalloc(&global_mem, global_mem_size), "cudaMalloc global");
	ErrorCheck(cudaMemcpy(global_mem, buffer, global_mem_size, cudaMemcpyHostToDevice),
			"cudaMemcpy device->host");

	ErrorCheck(cudaMalloc(&transblock_buffers, transblock_buffers_size), "cudaMalloc buffers");
	ErrorCheck(cudaMemset(transblock_buffers, 0, transblock_buffers_size), "cudaMemset buffers");
	ErrorCheck(cudaMalloc(&buffer_flags, buffer_flags_size), "cudaMalloc buffer-flags");
	ErrorCheck(cudaMemset(buffer_flags, 0, buffer_flags_size), "cudaMemset buffer-flags");

	ParticleSort<<<grid, block>>>(global_mem, transblock_buffers, buffer_flags, size);
	cudaThreadSynchronize();
	ErrorCheck(cudaGetLastError(), "kernel execution");
	
	ErrorCheck(cudaMemcpy(buffer, global_mem, global_mem_size, cudaMemcpyDeviceToHost),
			"cudaMemcpy host->device");
	ErrorCheck(cudaFree(global_mem), "cudaFree global");
	ErrorCheck(cudaFree(transblock_buffers), "cudaFree buffers");
	ErrorCheck(cudaFree(buffer_flags), "cudaFree buffer-flags");
}

static void ErrorCheck (cudaError_t cerr, const char *str)
{
	if (cerr == cudaSuccess) return;
	fprintf(stderr, "CUDA Runtime Error: %s\n at %s\n", cudaGetErrorString(cerr), str);
	exit(EXIT_FAILURE);
}

/// MAIN //////////////////////////////////////////////////////////////////////////////////
int main (int argc, char **argv)
{
	unsigned long elapsed = TestHarness(sort);
	fprintf(stderr, "Sort complete; time elapsed: %lu ms\n", elapsed);
	exit(EXIT_SUCCESS);
}

#endif
