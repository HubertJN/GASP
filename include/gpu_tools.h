#ifndef GPU_TOOLS_H
#define GPU_TOOLS_H

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>


// Prototype for function which initialises gpu
int gpuInitDevice(int deviceIndex, int* gpu_nsms);

// Function which copies model from host to device and initialises it
void gpuInitGrid(int L, int ngrids, int threadsPerBlock, int *ising_grids, int** d_ising_grids, int** d_neighbour_list);

// Function which initialised RNG on the GPU
void gpuInitRand(int ngrids, int threadsPerBlock, unsigned long rngseed,  curandState** d_state);


// Macro for error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Prototype for gpuAssert
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

// Prototype for RNG initialisation kernel
__global__ void init_gpurand(unsigned long long seed, int ngrids, curandState *state);

// For testing generation of random numbers on each thread
__global__ void populate_random(int length, float *rnd_array, curandState *state);

// Selects which gpu_method to use automatically //
int select_gpu_method(int L, int ngrids, int threadsPerBlock, int gpu_device );

#endif
