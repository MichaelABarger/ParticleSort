#ifndef PARTICLE_SORT_CU
#define PARTICLE_SORT_CU

#include <cuda.h>
#include <stdio.h>
#include "../testharness/testharness.h"

#define LIFESPAN 100


enum particleState { ALIVE, DYING, DEAD };


__device__ unsigned short *global_mem;
__device__ unsigned int *sums;
__device__ int dead_ct = 0;

__device__ void Collide (signed char *, signed char, short *, enum particleState *);
__device__ void Pass (short *, enum particleState *);
__device__ void Revive (signed char *, short *, enum particleState *, const signed char *);
__device__ void Die (signed char *, enum particleState *);


#define COLLIDE(v) Collide(&velocity,v,&momentum,&state)
#define PASS Pass(&momentum,&state)
#define REVIVE Revive(&velocity,&momentum,&state,&direction)
#define DIE Die(&velocity,&state)
__global__ void ParticleSort (unsigned short *global_mem, unsigned int *sums, unsigned long size)
{
	//extern __shared__ int shmem [];
	unsigned int *end = sums + size - 1;
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadID == 0)
		dead_ct = 0;
	signed char direction = -1;

	/* slot initialization */
	unsigned int *here = sums + threadID;

	/* particle initialization */
	enum particleState state = ALIVE;
	signed char velocity = 1 - (threadID & 0x01) * 2;
	unsigned int *position = here;
	unsigned int value = global_mem[threadID] + 1;
	short momentum = LIFESPAN;

	/* main sorting loop */
	do {
		/* prepare */
		*here = 0;
		direction = -direction;
		__syncthreads();

		/* move position if velocity is same as direction */
		/* perform wall collisions */
		if ((state != DEAD) && (velocity == direction)) {
			position += velocity;
			if (position < sums) {
				position = sums;
				COLLIDE(-velocity);
			} else if (position > end) {
				position = end;
				COLLIDE(-velocity);
			}
		}

		/* prepare collisions */
		char notfirst = atomicXor(position, value);
		__syncthreads();
		
		/* resolve collisions */
		unsigned int slotval = *position;
		unsigned int others = slotval ^ value;
		if (state == DEAD) {
			if ((value == others) && (slotval == 0) && notfirst) REVIVE;
			if ((value > others) != (direction > 0)) REVIVE; //logical XOR
		} else {
			if (others == 0) {
				if ((state == DYING) && !notfirst) DIE;
			}
			else if (value == others)
			       	if (!slotval && notfirst) COLLIDE(-direction);
			else if ((value > others) != (velocity > 0)) COLLIDE(-velocity); //logical XOR
			//else PASS;
		}
		__syncthreads();

	} while (dead_ct < size); 
	__syncthreads();

	/* we're done. copy everything back into global memory */
	atomicExch(position, value);
	__syncthreads();
	global_mem[threadID] = (unsigned short)*here - 1;
}

__device__ void Collide (signed char *velocity, signed char v, short *momentum, enum particleState *state)
{
	*velocity = v;
	if (--(*momentum) <= 0) {
		*state = DYING;
		*momentum = 0;
	}
}

__device__ void Pass (short *momentum, enum particleState *state)
{
	*momentum = min(*momentum + 1, LIFESPAN);
	*state = ALIVE;
}

__device__ void Revive (signed char *velocity, short *momentum, enum particleState *state, const signed char *direction)
{
	*momentum = 1;
	*velocity = -(*direction);
	*state = DYING;
	atomicAdd(&dead_ct, -1);
}

__device__ void Die (signed char *velocity, enum particleState *state)
{
	*velocity = 0;
	*state = DEAD;
	atomicAdd(&dead_ct, 1);
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
	ErrorCheck(cudaMalloc(&sums, size * sizeof(unsigned int)), "cudaMalloc sums");
	
	ErrorCheck(cudaMemcpy(global_mem, buffer, size * sizeof(unsigned short), cudaMemcpyHostToDevice), "cudaMemcpy host->device global");

	dim3 grid (1, 1, 1);
	dim3 block (size, 1, 1);
	/*size_t shmem_size = size * 4;*/
	ParticleSort<<<grid, block>>>(global_mem, sums, size);

	ErrorCheck(cudaMemcpy(buffer, global_mem, size * sizeof(unsigned short), cudaMemcpyDeviceToHost), "cudaMemcpy device->host");

	ErrorCheck(cudaFree(global_mem), "cudaFree global");
	ErrorCheck(cudaFree(sums), "cudaFree sums");
}

int main (int argc, char **argv)
{
	unsigned long elapsed = TestHarness(sort);
	fprintf(stderr, "Sort complete; time elapsed: %lu ms\n", elapsed);
	exit(EXIT__SUCCESS);
}

#endif
