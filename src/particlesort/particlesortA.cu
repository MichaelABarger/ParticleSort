#ifndef PARTICLE_SORT_CU
#define PARTICLE_SORT_CU

#include <cuda.h>
#include <stdio.h>
#include "../testharness/testharness.h"

#define LIFESPAN 50


enum particleState { ALIVE, DYING, DEAD };


__device__ unsigned short *global_mem;
__device__ int idle_ct = 0;

__device__ void Collide (signed char *, char *, enum particleState *);
__device__ void Pass (char *, enum particleState *);
__device__ void Revive (signed char *, char *, enum particleState *, const signed char *);
__device__ void Die (enum particleState *);



#define COLLIDE Collide(&velocity,&momentum,&state)
#define PASS Pass(&momentum,&state)
#define REVIVE Revive(&velocity,&momentum,&state,&direction)
#define DIE Die(&state)
__global__ void ParticleSort (unsigned short *global_mem, unsigned long size)
{
	extern __shared__ int shmem [];
	int *end = &shmem[size - 1];
	int threadID = blockDim.x * blockIdx.x + threadIdx.x;

	signed char direction = -1;

	/* particle initialization */
	enum particleState state = ALIVE;
	signed char velocity = 1 - (threadID & 0x01) * 2;
	int *position = shmem + threadID;
	unsigned int value = global_mem[threadID];
	char momentum = LIFESPAN;

	/* slot initialization */
	int *here = position;

	/* main sorting loop */
	do {
		/* prepare */
		*here = 0;
		direction = -direction;
		__syncthreads();

		/* move position if velocity is same as direction */
		/* perform wall collisions */
		if (state != DEAD && velocity == direction) {
			position += velocity;
			if (position < shmem) {
				position = shmem;
				COLLIDE;
			} else if (position > end) {
				position = end;
				COLLIDE;
			}
		}

		/* resolve collisions */
		*position += ((state != DEAD) ? velocity : -direction) * value;
		__syncthreads();
		
		int sum = *position;
		int abs_sum = abs(sum);

		switch (state) {

		case DEAD:
			if (abs_sum < value)
				REVIVE;
			break;

		case DYING:
			if (abs_sum == value) {
				DIE;
				break;
			} /* fall through if not */

		case ALIVE:
			if ((abs_sum == value) || (sum == 0))
				break;
			if (sum < 0) 
				COLLIDE;
			else 
				PASS;
		}
		__syncthreads();

	} while (idle_ct < size);

	/* we're done. copy everything back into global memory */
	*position = value;
	__syncthreads();
	global_mem[threadID] = *here;
}

__device__ void Collide (signed char *velocity, char *momentum, enum particleState *state)
{
	*velocity = -(*velocity);
	if (--(*momentum) <= 0)
		*state = DYING;
}

__device__ void Pass (char *momentum, enum particleState *state)
{
	*momentum = min(*momentum + 1, LIFESPAN);
	*state = ALIVE;
}

__device__ void Revive (signed char *velocity, char *momentum, enum particleState *state, const signed char *direction)
{
	*momentum = 1;
	*velocity = -(*direction);
	*state = ALIVE;
	atomicAdd(&idle_ct, -1);
}

__device__ void Die (enum particleState *state)
{
	*state = DEAD;
	atomicAdd(&idle_ct, 1);
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
	ErrorCheck(cudaMalloc(&global_mem, size * sizeof(unsigned short)), "cudaMalloc global");
	
	ErrorCheck(cudaMemcpy(global_mem, buffer, size * sizeof(unsigned short), cudaMemcpyHostToDevice), "cudaMemcpy host->device global");

	dim3 grid (1, 1, 1);
	dim3 block (size, 1, 1);
	size_t shmem_size = size * 4;
	ParticleSort<<<grid, block, shmem_size>>>(global_mem, size);

	ErrorCheck(cudaMemcpy(buffer, global_mem, size * 2, cudaMemcpyDeviceToHost), "cudaMemcpy device->host");

	ErrorCheck(cudaFree(global_mem), "cudaFree global");
}

int main (int argc, char **argv)
{
	unsigned long elapsed = TestHarness(sort);
	fprintf(stderr, "Sort complete; time elapsed: %lu ms\n", elapsed);
	exit(EXIT__SUCCESS);
}

#endif
