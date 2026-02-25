#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>

extern "C" {
#include "mc_cpu.h"       // Extend this for GPU version
}

#include "gpu_tools.h"

typedef struct {
  int L;
  int ngrids;
  int* ising_grids;
  int* d_ising_grids;     // Device copy of ising_Grids
  int* d_neighbour_list;  // Neighbour list array
} mc_gpu_grids_t;

typedef struct {
  curandState* d_state;   // GPU RNG state
  int threadsPerBlock;    // Threads to launch per thread block
  int gpu_method;         // GPU method to use
} gpu_run_t;


// pre-compute acceptance probabilities for spin flips
void preComputeProbs_gpu(double beta, double h);

// pre-compute acceptance probabilities for Kawasaki NN swaps
void preComputeProbsKawasaki_4NN_gpu(double beta, double h);

// pre-compute neighbours
void preComputeNeighbours_gpu(const int L, int *d_ising_grids, int *d_neighbour_list);


// MC sweep on the GPU - 3 versions
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitmap32(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);
__global__ void mc_sweep_gpu_bitmap64(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps);

// Compute magnetisation on the GPU for all grids in parallel
__global__ void compute_magnetisation_gpu(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation);

// Compute largest cluster size of sites with spin=spin for all grids in parallel
__global__ void compute_largest_cluster_gpu(const int L, const int ngrids, int *d_ising_grids, const int spin, int *d_work, int *lclus_size);

// Main driver function on GPU
void mc_driver_gpu(mc_gpu_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, gpu_run_t gpu_state, GridOutputFunc func);


// Neighbour list squeezed into constant memory. We can have 64k max in constant memory.
// Limited to MAXL = 256 by upper limit of what can be stored in uint8_t. 
// Limited to MAXL = 65,535 by upper limit of what can be stored in uint16_t. 
// But constant memory is limited to 64k, so each array will be 32k max meaning
//     - If using uint16_t MAXL limited to 16,384 by constant memory size limit
//     - If using uint8_t MAXL limited to 65,535 by constant memory size limit
// Using uint16_t and MAXL = 8192 should cover all use cases and use only half constant memory


#define MAXL 8192
__constant__ uint16_t dc_next[MAXL]; // Index of positive + 1 shift neighbour
__constant__ uint16_t dc_prev[MAXL]; // Index of negative - 1 shift neighbour

