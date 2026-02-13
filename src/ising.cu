// -*- mode: C -*-
/* ==========================================================================================
                                 GPU_2DIsing.cu

Implementation of the 2D Ising model in CUDA. Each CUDA thread simulates an independent 
instance of the 2D Ising model in parallel with an independent random number sequence. Draws
heavily from the work of Weigel et al, [J. Phys.: Conf. Ser.921 012017 (2017)] but used here
for gathering rare event statistics on nucleation during magnetisation reversal. 
 ===========================================================================================*/
// D. Quigley. Univeristy of Warwick

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>

extern "C" {
  #include "io.h"
  #include "grid.h"
}

#include "mc_gpu.h"
#include "gpu_tools.h"


bool run_gpu = true;    // Run using GPU
bool run_cpu = false;   // Run using CPU

int gpu_nsms;           // Number of multiprocessors on GPU

int main (int argc, char *argv[]) {

/*=================================
   Constants and variables
  =================================*/ 
  
  int L       = 64;            // Size of 2D Ising grid. LxL grid squares.
  int ngrids  = 1;             // Number of replicas of 2D grid to simulate
  int tot_nsweeps = 100;       // Total number of MC sweeps to simulate on each grid

  int itask = 0;               // 0 = count samples which nucleate, 1 = compute committor

  int mag_output_int  = 100;   // Number of MC sweeps between calculation of magnetisation
  int grid_output_int = 1000;  // Number of MC sweeps between dumps of grid to file

  double beta = 0.54;          // Inverse temperature
  double h = 0.07;             // External field

  double dn_threshold = -0.90;   // Magnetisation at which we consider the system to have reached spin up state
  double up_threshold =  0.90;   // Magnetisation at which we consider the system to have reached spin down state

  //unsigned long rngseed = 2894203475;  // RNG seed (fixed for development/testing)
  unsigned long rngseed = (long)time(NULL);

  int threadsPerBlock = 32;            // Number of threads/replicas to run in each threadBlock
  int gpu_device = -1;                 // GPU device to use
  int gpu_method = 0;                  // Which MC sweep kernel to use

/*=================================
   Process command line arguments 
  =================================*/ 
  if (argc != 11) {
    printf("Usage : GPU_2DIsing nsweeps nreplicas mag_output_int grid_output_int threadsPerBlock gpu_device gpu_method beta h itask \n");
    exit(EXIT_FAILURE);
  }

  tot_nsweeps     = atoi(argv[1]);  // Number of MC sweeps to simulate
  ngrids          = atoi(argv[2]);  // Number of replicas (grids) to simulate
  mag_output_int  = atoi(argv[3]);  // Sweeps between printing magnetisation
  grid_output_int = atoi(argv[4]);  // Sweeps between dumps of grid
  threadsPerBlock = atoi(argv[5]);  // Number of thread per block (multiple of 32)
  gpu_device      = atoi(argv[6]);  // Which GPU device to use (normally 0) 
  gpu_method      = atoi(argv[7]);  // Which kernel to use for MC sweeps
  beta            = atof(argv[8]);  // Inverse temperature
  h               = atof(argv[9]);  // Magnetic field
  itask           = atof(argv[10]); // Calculation task

  char cv[] = "magnetisation";             // Collective variable to use in determining fate

/*=================================
   Delete old output 
  ================================*/
  remove("gridstates.bin");


/*=================================
   Write output header 
  ================================*/
  if (itask==0) {
    printf("# isweep    nucleated fraction\n");
  }

/*=================================
   Initialise simulations
  =================================*/ 

  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)

  float result;     // result of calculation
  
  if (itask==0) {   // counting nucleated samples over time

    // Initialise as 100% spin down for all grids
    ising_grids = init_grids_uniform(L, ngrids, -1);
    grid_fate = NULL ; // not used
    
  } else if (itask==1) {

    // Read from gridinput.bin
    ising_grids = init_grids_from_file(L, ngrids);

    // Initialise states
    grid_fate = init_fates(ngrids);
    
  } else {

    ising_grids = NULL;
    grid_fate = NULL;
    printf("Error - unknown value of itask!");
    exit(EXIT_FAILURE);

  }


  if (run_gpu==true) {

    // Initialise GPU device(s)
    int igpu = gpuInitDevice(gpu_device, &gpu_nsms); 
    if (igpu==-1){
      printf("Falling back to CPU\n");
      run_cpu=true;
      run_gpu=false;
    }
    
  }

/*=================================
    Run simulations - CPU version
  =================================*/ 

  if (run_cpu==true) {


    // Initialise host RNG
    init_genrand(rngseed);

    // Precompute acceptance probabilities for flip moves
    preComputeProbs_cpu(beta, h);

    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = itask; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = &result;

    // Perform the MC simulations
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, write_ising_grids); 

  }

/*=================================
    Run simulations - GPU version
  =================================*/ 
  if (run_gpu==true){


    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    // Initialise model grid on GPU
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    curandState *d_state;                  // Pointer to device RNG states
   
    // Iniialise RNG on GPU
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state); 
      
    // Precompute acceptance probabilities for flip moves
    preComputeProbs_gpu(beta, h);


    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = itask; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = &result;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, write_ising_grids);
    
    // Free device arrays
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

    
 }


/*=================================================
    Tidy up memory used in both GPU and CPU paths
  =================================================*/ 
  free(ising_grids);

  return EXIT_SUCCESS;

}
