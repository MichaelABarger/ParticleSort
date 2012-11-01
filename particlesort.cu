#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "testharness.h"

__device__ unsigned short *global_mem;


__global__ void ParticleSort (unsigned short *global_mem, unsigned long total_size)
{
	/* REGISTER INITIALIZATION */
	extern __shared__ int shared [];
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned short particle_value = global_mem[threadID];
	int *particle_position, *slot;
        particle_position = slot = shared + 1 + threadID;
	float particle_velocity = 1.0f - (particle_value & 0x0001) * 2.0f;

	int is_idle = 0;


	/* SHARED MEMORY INITIALIZATION */
	if (threadID == 0)
		*shared = 0;

	/* MAIN LOOP */
	do {
		/* set slot value to 0 */
		*slot = 0;

		/* move non-idle particles */
		if (!is_idle) {
			if (particle_velocity < 0.3f) { /* idle particles that are too slow */
				/* (void) atomicAdd(shared, 1);*/
				(*shared)++;
				particle_velocity = 0.3f;
				is_idle = 1;
			} else { 
				/* move particles that still have velocity */
				particle_position += __float2int_ru(particle_velocity);
				if (particle_position < shared + 1) {
					particle_position = shared + 1;// do something
					particle_velocity *= -0.8f;	
				} else if (particle_position > shared + 2 * blockDim.x + 1) {
					particle_position = shared + 2 * blockDim.x + 1;// do something
					particle_velocity *= -0.8f;	
				}
			}
		}

		/* add particle's value to current-position slot's running sum */
		/* this happens whether idle or not */
		*particle_position += copysignf(particle_value, particle_velocity);
		/*(void) atomicAdd(particle_position, copysignf(particle_value, particle_velocity));*/
		__syncthreads();

		/* do collisions */
		particle_velocity *= copysignf(particle_value, particle_velocity) / *slot;
		if (is_idle && particle_velocity > 0.3f) {
			(*shared)--;
			/*(void) atomicSub(shared, 1);*/
			is_idle = 0;
		}
		__syncthreads();
	} while (*shared < blockDim.x);

	/* END OF LIFE CLEAN-UP */
	*particle_position = particle_value;
	__syncthreads();
	global_mem[threadID] = *slot;
}

__global__ void ParticleSort2(unsigned short *global_mem, unsigned long total_size)
{
	global_mem[blockDim.x * blockIdx.x + threadIdx.x] = 5;
}

void ErrorCheck (cudaError_t cerr, const char *str)
{
	if (cerr == cudaSuccess) 
		return;
	fprintf(stderr, "CUDA Runtime Error: %s\n at %s\n", cudaGetErrorString(cerr), str);
	exit(-1);
}


extern void sort (unsigned short *buffer, unsigned long size)
{
	ErrorCheck(cudaMalloc(&global_mem, size * 2), "cudaMalloc");
	
	ErrorCheck(cudaMemcpy(global_mem, buffer, size * 2, cudaMemcpyHostToDevice), "cudaMemcpy host->device");

	dim3 grid (1, 1, 1);
	dim3 block (size, 1, 1);
	size_t shm_size = (size + 1) * 4;
	ParticleSort<<<grid, block, shm_size>>>(global_mem, size);
	ErrorCheck(cudaGetLastError(), "kernel execution");


	ErrorCheck(cudaMemcpy(buffer, global_mem, size * 2, cudaMemcpyDeviceToHost), "cudaMemcpy device->host");
	ErrorCheck(cudaFree(global_mem), "cudaFree");
}

int main (int argc, char **argv)
{
	unsigned long elapsed = TestHarness(sort);
	fprintf(stderr, "Sort complete; time elapsed: %lu ms\n", elapsed);
	exit(EXIT__SUCCESS);
}
