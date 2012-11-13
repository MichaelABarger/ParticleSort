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

#define BLOCK 512
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
struct particle {
	unsigned int color;
	unsigned char momentum;
};

static __device__ void ReadParticle (const unsigned int, struct particle *);
static __device__ void WriteParticle (const struct particle *, volatile unsigned int *);
static __device__ void Collide (struct particle *, struct particle *);
static __device__ void Bump (struct particle *, unsigned int *);
static __device__ void Reside (struct particle *, unsigned int *);
static __device__ void Swap (struct particle *, struct particle *);

#ifdef DEBUG
	__device__ unsigned int collision_counter = 0;
	__device__ unsigned int bump_counter = 0;
	__device__ unsigned int pass_counter = 0;
	__device__ unsigned int reside_counter = 0;
	__device__ unsigned int i_counter = 0;
	struct slot_state {
		struct particle pre_right;
		struct particle pre_left;
		unsigned int pre_resident;
		struct particle post_right;
		struct particle post_left;
		unsigned int post_resident;
	};
#define SLOT_CT 4
#define ITER_CT 100
	__device__ struct slot_state slots[SLOT_CT][ITER_CT];
#endif

extern "C" __global__ void ParticleSort (unsigned int *global_mem,
					 unsigned long size)
{
	/* define shared memory */
	volatile __shared__ unsigned int beginning [BLOCK];
	volatile __shared__ unsigned int isNotComplete;


	/* define registers */
	const int absThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	struct particle going_left, going_right;

	enum {BEGINNING, MIDDLE, END} role;
	if (threadIdx.x == 0) role = BEGINNING;
	else if (threadIdx.x == min(blockDim.x - 1, (int)size - 1)) role = END;
	else role = MIDDLE;

	volatile unsigned int *const here = beginning + threadIdx.x;

	unsigned int resident;
	signed char i = 0;




	/* initial coalesced global memory read */
	resident = MOMENTUM_INIT | (global_mem[absThreadID] + 1);
	if (threadIdx.x & 0x01 || role == END) {
		ReadParticle(resident, &going_left);
		resident = 0;
	}
	switch (role) {
	case BEGINNING:
		*here = 0;
		// fall through
	case MIDDLE:
		*(here + 1) = resident;
	}
	resident = 0;
	__syncthreads();

	/* sorting loop */
	do {
		if (role == BEGINNING)
			isNotComplete = FALSE;

		// non-diverging conditional
		if (i & 0x01) { // if moving left

			ReadParticle(*here, &going_left);

#ifdef DEBUG
			if ((threadIdx.x < SLOT_CT) && (i < ITER_CT)) {
				slots[threadIdx.x][i].pre_right = going_right;
				slots[threadIdx.x][i].pre_left = going_left;
				slots[threadIdx.x][i].pre_resident = resident;
			}
#endif

			if (going_left.color) {
				if (going_right.color)
					Collide(&going_left, &going_right);
				if (resident) {
					if (going_left.color > resident)
						Bump(&going_left, &resident);
				} else {
					if (!going_right.color && !going_left.momentum)
						Reside(&going_left, &resident);
				}
			}

#ifdef DEBUG
			if ((threadIdx.x < SLOT_CT) && (i < ITER_CT)) {
				slots[threadIdx.x][i].post_right = going_right;
				slots[threadIdx.x][i].post_left = going_left;
				slots[threadIdx.x][i].post_resident = resident;
			}
#endif

			__syncthreads();
			// prepare for moving right
			switch (role) {
			case BEGINNING:
				if (going_left.color)
					DECREASE_MOMENTUM(going_left);
				WriteParticle(&going_left, here);
				RESET(going_left);
				// fall through
			case MIDDLE:
				WriteParticle(&going_right, here + 1);
			}
		} else { // if moving right

			ReadParticle(*here, &going_right);

#ifdef DEBUG
			if ((threadIdx.x < SLOT_CT) && (i < ITER_CT)) {
				slots[threadIdx.x][i].pre_right = going_right;
				slots[threadIdx.x][i].pre_left = going_left;
				slots[threadIdx.x][i].pre_resident = resident;
			}
#endif

			if (going_right.color) {
				if (going_left.color)
					Collide(&going_left, &going_right);
				if (resident) {
					if (going_right.color < resident)
						Bump(&going_right, &resident);
				} else {
					if (!going_left.color && !going_right.momentum)
						Reside(&going_right, &resident);
				}
			}


#ifdef DEBUG
			if ((threadIdx.x < SLOT_CT) && (i < ITER_CT)) {
				slots[threadIdx.x][i].post_right = going_right;
				slots[threadIdx.x][i].post_left = going_left;
				slots[threadIdx.x][i].post_resident = resident;
			}
#endif
			__syncthreads();
			// prepare for moving left
			switch (role) {
			case END:
				if (going_right.color)
					DECREASE_MOMENTUM(going_right);
				WriteParticle(&going_right, here);
				RESET(going_right);
				// fall through
			case MIDDLE:
				WriteParticle(&going_left, here - 1);
			}
		}

		++i;
		if (!resident)
			isNotComplete = TRUE;
		__syncthreads();
	} while (isNotComplete);

#ifdef DEBUG
	if (role == END)
		i_counter = i;
#endif

	/* read sorted values back to array */
	global_mem[absThreadID] = ((resident - 1) & COLOR_MASK);
}

static __device__ void ReadParticle (const unsigned int src, struct particle *dest)
{
	dest->momentum = src >> COLOR_WIDTH;
	dest->color = src & COLOR_MASK;
}

static __device__ void WriteParticle (const struct particle *src, volatile unsigned int *dest)
{
	*dest = (src->momentum << COLOR_WIDTH) | src->color;
}

static __device__ void Collide (struct particle *L, struct particle *R)
{
	if (L->color < R->color) {
#ifdef DEBUG
		atomicAdd(&pass_counter, 1);
#endif
		INCREASE_MOMENTUM_PTR(L);
		INCREASE_MOMENTUM_PTR(R);
	} else {
#ifdef DEBUG
		atomicAdd(&collision_counter, 1);
#endif
		DECREASE_MOMENTUM_PTR(L);
		DECREASE_MOMENTUM_PTR(R);
		Swap(L, R);
	}
}

static __device__ void Bump (struct particle *incoming, unsigned int *resident)
{
#ifdef DEBUG
	atomicAdd(&bump_counter, 1);
#endif 
	unsigned int temp = incoming->color;
	incoming->color = *resident;
	DECREASE_MOMENTUM_PTR(incoming);
	*resident = temp;
}

static __device__ void Reside (struct particle *incoming, unsigned int *resident)
{
#ifdef DEBUG
	atomicAdd(&reside_counter, 1);
#endif
	*resident = incoming->color;
	incoming->color = 0;
}

static __device__ void Swap (struct particle *L, struct particle *R)
{
		L->color ^= R->color;
		R->color ^= L->color;
		L->color ^= R->color;
		L->momentum ^= R->momentum;
		R->momentum ^= L->momentum;
		L->momentum ^= R->momentum;
}

/// CUDA HOST /////////////////////////////////////////////////////////////////////////////
static void ErrorCheck (cudaError_t cerr, const char *str);
__device__ unsigned int *global_mem;

extern "C" void sort (unsigned int *buffer, unsigned long size)
{
	dim3 grid (1);
	dim3 block (size);
	size_t transfer_size = size * sizeof(int);

	ErrorCheck(cudaMalloc(&global_mem, transfer_size), "cudaMalloc global");
	ErrorCheck(cudaMemcpy(global_mem, buffer, transfer_size, cudaMemcpyHostToDevice),
			"cudaMemcpy device->host");

	ParticleSort<<<grid, block>>>(global_mem, size);

	ErrorCheck(cudaMemcpy(buffer, global_mem, transfer_size, cudaMemcpyDeviceToHost),
			"cudaMemcpy host->device");
	ErrorCheck(cudaFree(global_mem), "cudaFree global");

#ifdef DEBUG
	unsigned int collisions, bumps, passes, resides, iters;
	struct slot_state slots[SLOT_CT][ITER_CT];
	ErrorCheck(cudaMemcpyFromSymbol(&collisions, "collision_counter", sizeof(int), 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol collision_counter");
	ErrorCheck(cudaMemcpyFromSymbol(&bumps, "bump_counter", sizeof(int), 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol bump_counter");
	ErrorCheck(cudaMemcpyFromSymbol(&passes, "pass_counter", sizeof(int), 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol pass_counter");
	ErrorCheck(cudaMemcpyFromSymbol(&resides, "reside_counter", sizeof(int), 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol reside_counter");
	ErrorCheck(cudaMemcpyFromSymbol(&iters, "i_counter", sizeof(int), 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol i_counter");
	fprintf(stderr, "*** DEBUG ***\n // ITERATIONS: %u\n // COLLISIONS: %u\n // BUMPS: %u\n // PASSES: %u\n // RESIDES: %u\n",
			iters, collisions, bumps, passes, resides);
	
	ErrorCheck(cudaMemcpyFromSymbol(slots, "slots", sizeof(struct slot_state) * SLOT_CT * ITER_CT, 0, cudaMemcpyDeviceToHost),
			"cudaMemcpyFromSymbol slots");
	for (int i = 0; i < min(ITER_CT, iters); i++) {
		fprintf(stderr, "    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");
		fprintf(stderr, "    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");

		fprintf(stderr, "\n    |");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].pre_right.color);
		fprintf(stderr, "    |");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].post_right.color);

		fprintf(stderr, "\n    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");
		fprintf(stderr, "    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");

		fprintf(stderr, "\n%3i |", i);
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].pre_resident);
		fprintf(stderr, " -> |", i);
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].post_resident);

		fprintf(stderr, "\n    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");
		fprintf(stderr, "    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");

		fprintf(stderr, "\n    |");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].pre_left.color);
		fprintf(stderr, "    |");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "%5u|", slots[j][i].post_left.color);

		fprintf(stderr, "\n    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");
		fprintf(stderr, "    +");
		for (int j = 0; j < SLOT_CT; j++)
			fprintf(stderr, "-----+");
		fprintf(stderr, "\n\n");
	}
#endif
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
