#include <stdlib.h>
#include <math.h>
#include "mt19937ar.h"   // External RNG for CPU reference implementation

typedef struct {
  int  L;                // Size of each L x L grid
  int  ngrids;           // Number of grids
  int* ising_grids;      // Pointer to first entry in first grid
} mc_grids_t;

typedef struct {
  int tot_nsweeps;     // Max number of MC sweeps to perform 
  int mag_output_int;  // Interval at which to calculate magnetisation
  int grid_output_int; // Interval at which to write grids to file
} mc_sampler_t;

typedef struct {
  int itask;         // What to calculate?
  double dn_thr;     // CV value below which lies the down macrostate
  double up_thr;     // CV value above which lies the up macrostate
  int ninputs;       // Number of input grids
  int initial_spin;  // Spin of the parent phase we're nucleating from
  char* cv;          // Collective variable on which thresholds are defined
  float* result;     // Pointer to result array 
  char* filename;    // Filename for output of grids to disk
  char* dynamics;    // "kawasaki" or "spin_flip"
} mc_function_t;



// Function typedef obeyed by functions which output/store grids
typedef int (*GridOutputFunc)(int L, int ngrids, int* grid_data, int isweep, float* magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr, char *filename);

// pre-compute acceptance probabilities for spin flips
void preComputeProbs_cpu(double beta, double h);

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps);

// Compute magnetisation on the cpu
void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, float *magnetisation);

// Compute size of largest cluster with spin=spin for grid with index grid_index
void compute_largest_cluster_cpu(int L, int* ising_grids, const int grid_index, int spin, float *lclus_size);

// Main driver routine
void mc_driver_cpu(mc_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, GridOutputFunc func);
