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

extern "C" __global__ void ParticleSort (unsigned int *global_mem,
					 unsigned int *transblock_buffers,
					 unsigned int *buffer_flags,
					 unsigned long size)
{
	/* define shared memory */
	volatile __shared__ unsigned int beginning [BLOCK];
	__shared__ unsigned int left_incoming [BUFFER_SIZE];
	__shared__ unsigned int left_outgoing [BUFFER_SIZE];
	__shared__ unsigned int right_incoming [BUFFER_SIZE];
	__shared__ unsigned int right_outgoing [BUFFER_SIZE];
	__shared__ unsigned int *cur_left_incoming, *cur_left_outgoing,
		 			 *cur_right_incoming, *cur_right_outgoing;
	__shared__ unsigned int isNotComplete;


	/* define registers */
	const int absThreadID = blockIdx.x * blockDim.x + threadIdx.x;
	
	struct particle going_left, going_right;

	enum {BEGINNING, LEFT, MIDDLE, RIGHT, END, IDLE} role;
	if (absThreadID == 0) role = BEGINNING;
	else if (absThreadID == size - 1) role = END;
	else if (absThreadID >= size) role = IDLE;
	else if (threadIdx.x == 0) role = LEFT;
	else if (threadIdx.x == blockDim.x - 1) role = RIGHT;
	else role = MIDDLE;

	volatile unsigned int *const here = beginning + threadIdx.x;

	unsigned int resident;
	signed char i = 0;


	/* initial coalesced global memory read */
	if (role != IDLE) {
		resident = MOMENTUM_INIT | (global_mem[absThreadID] + 1);
		if (threadIdx.x & 0x01 || role == END) {
			ReadParticle(resident, &going_left);
			resident = 0;
		}
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
			if (role != IDLE) {
				ReadParticle(*here, &going_left);

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
			}
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
			if (role != IDLE) {
				ReadParticle(*here, &going_right);

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
			}
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
		if ((role != IDLE) && !resident)
			isNotComplete = TRUE;
		__syncthreads();
	} while (isNotComplete);

	/* read sorted values back to array */
	if (role != IDLE)
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
		INCREASE_MOMENTUM_PTR(L);
		INCREASE_MOMENTUM_PTR(R);
	} else {
		DECREASE_MOMENTUM_PTR(L);
		DECREASE_MOMENTUM_PTR(R);
		Swap(L, R);
	}
}

static __device__ void Bump (struct particle *incoming, unsigned int *resident)
{
	unsigned int temp = incoming->color;
	incoming->color = *resident;
	DECREASE_MOMENTUM_PTR(incoming);
	*resident = temp;
}

static __device__ void Reside (struct particle *incoming, unsigned int *resident)
{
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
