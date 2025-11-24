// -*- mode: C -*-

#include <stdio.h>
#include "mc_gpu.h"

#ifdef PYTHON
#include <Python.h>
#endif

extern "C" {
   #include "io.h"
}
  
// Cache of acceptance probabilities 
__constant__ float d_Pacc[20];   // gpu constant memory

// Dynamic shared memory for storage of bits
extern __shared__ uint8_t shared_grid[];

// populate acceptance probabilities
void preComputeProbs_gpu(double beta, double h) {

    float *h_Pacc=(float *)malloc(20*sizeof(float));

    int s, nsum, index;  
    for (s=-1;s<2;s=s+2){
      for (nsum=-4;nsum<5;nsum=nsum+2){
        index = 5*(s+1) + nsum + 4;
        h_Pacc[index] = 2.0f*(float)s*((float)nsum+(float)h);
        h_Pacc[index] = expf(-(float)beta*h_Pacc[index]); // single precision version for GPU
      }
    }
  
    gpuErrchk( cudaMemcpyToSymbol(d_Pacc, h_Pacc, 20*sizeof(float),0, cudaMemcpyHostToDevice ) );
    free(h_Pacc);

  }  

void preComputeNeighbours_gpu(const int L, int *d_ising_grids, int *d_neighbour_list){

// These could probably be cached in shared memory since they are the same for all threads.

int *h_neighbour_list = (int *)malloc(L*L*4*sizeof(int));

int spin_index;
for (spin_index=0;spin_index<L*L;spin_index++){

  int row = spin_index/L;
  int col = spin_index%L;

  h_neighbour_list[4*(row*L+col) + 0] = L*((row+1)%L) + col;
  h_neighbour_list[4*(row*L+col) + 1] = L*((row+L-1)%L) + col;
  h_neighbour_list[4*(row*L+col) + 2] = L*row + (col+1)%L;
  h_neighbour_list[4*(row*L+col) + 3] = L*row + (col+L-1)%L;

}

gpuErrchk( cudaMemcpy(d_neighbour_list, h_neighbour_list, 4*L*L*sizeof(int),cudaMemcpyHostToDevice ) );

free(h_neighbour_list); 

if (L>MAXL) {
  fprintf(stderr,"Error: L exceeds maximum limit\n");
  exit(EXIT_FAILURE);
}

/// Also store a version in constant memory
uint16_t *hc_next = (uint16_t *)malloc(L*sizeof(uint16_t));
uint16_t *hc_prev = (uint16_t *)malloc(L*sizeof(uint16_t));

for (spin_index=0;spin_index<L;spin_index++){

  hc_next[spin_index] = (spin_index+1)%L;
  hc_prev[spin_index] = (spin_index+L-1)%L;

}

gpuErrchk( cudaMemcpyToSymbol(dc_next, hc_next, L*sizeof(uint16_t),0, cudaMemcpyHostToDevice ) );
gpuErrchk( cudaMemcpyToSymbol(dc_prev, hc_prev, L*sizeof(uint16_t),0, cudaMemcpyHostToDevice ) );
  
free(hc_next); 
free(hc_prev);

}


// sweep on the gpu - default version
__global__ void mc_sweep_gpu(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;
  int index;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    int N = L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;

    // Pointer to local grid
    int *loc_grid = &d_ising_grids[idx*N]; // pointer to device global memory 


    int imove, my_idx, spin, n1, n2, n3, n4, row, col;
    for (imove=0;imove<N*nsweeps;imove++){

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));

      row = my_idx/L;
      col = my_idx%L;

      spin = loc_grid[my_idx];

      // find neighbours
      n1 = loc_grid[L*((row+1)%L) + col];
      n2 = loc_grid[L*((row+L-1)%L) + col];
      n3 = loc_grid[L*row + (col+1)%L];
      n4 = loc_grid[L*row + (col+L-1)%L];

      //n_sum = 4;
      index = 5*(spin+1) + n1+n2+n3+n4 + 4;

      // The store back to global memory, not the branch or the RNG generation
      // seems to be the killer here.
      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept
          loc_grid[my_idx] = -1*spin;
      } 
      
      // Try avoiding the branch entirely - this seems quite slow
      //diff = curand_uniform(&localState) - d_Pacc[index] ;
      //spin = spin * lrintf(copysignf(1.0f,diff)); 
      //loc_grid[my_idx] = spin;

      // This is even slower (and has a hidden branch)
      //diff = curand_uniform(&localState) - d_Pacc[index] ;
      //spin = signbit(diff) ? -1*spin : spin ;
      //loc_grid[my_idx] = spin;
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

  }

  return;

}

inline __device__ int bits_to_ints(uint8_t* grid, int index){
  // L must be the number of entries in each row here. Important if ever adapting to non-square grids.

  uint8_t one = 1U;
  int blookup[2] = {-1, 1};

  //nt ibyte = (index)/8;
  //int ibit  = (index)%8;

  // From CUDA-C best practices guide
  int ibyte = index >> 3;
  int ibit  = index & 7 ;

  return blookup[(grid[ibyte] >> ibit) & one]; 

}

// sweep on the gpu - packs a cache of the current grid into on-GPU shared memory
// for efficiency, and using a single bit represenation to acheive this. Mustn't
// be used if L*L*threadsPerBlock/8 > SHGRIDSIZE. 
__global__ void mc_sweep_gpu_bitrep(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  const int blookup[2] = {-1, 1};

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];

    // how many bytes do we need per thread to store L*L spins as single bytes
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes ++; }

    // If nbytes x threadsPerBlock is greater than the shared memory size
    // then we've got a problem, but should have had a kernel launch 
    // failure before getting this far so not checking that here.

    // Pointer to part of this storage used by the current thread
    uint8_t *loc_grid = &shared_grid[nbytes*threadIdx.x];
    uint8_t one  = 1U;
    uint8_t zero = 0U;

    // zero the local grid
    int ibyte;
    for (ibyte=0;ibyte<nbytes;ibyte++){ loc_grid[ibyte] = zero ; }

    // Fill this with the current state of the grid to be manipulated by this thread
    int ibit=0, spin;
    ibyte = 0;
    for (spin=0;spin<L*L;spin++){ 
        if ( d_ising_grids[L*L*idx + spin] == 1 ) {
          loc_grid[ibyte] |= one << ibit ;
        }
        ibit++;
        if (ibit==8) {
            ibit=0;
            ibyte++;
        }
    }
     

    float shrink = 1.0f - FLT_EPSILON;
    int imove, row, col, index,  n1, n2, n3, n4;
    uint16_t spin_index;

    for (imove=0;imove<L*L*nsweeps;imove++){

      spin_index = __float2uint_rd((float)L*L*shrink*curand_uniform(&localState));
      row = spin_index/L;
      col = spin_index%L;
      
      //spin = bits_to_ints(loc_grid, spin_index);
      spin = blookup[(loc_grid[spin_index >> 3] >> (spin_index & 7)) & one];

      // find neighbours
      n1 = L*((row+1)%L) + col;
      n1 = blookup[(loc_grid[n1 >> 3] >> (n1 & 7)) & one];

      n2 = L*((row+L-1)%L) + col;
      n2 = blookup[(loc_grid[n2 >> 3] >> (n2 & 7)) & one];

      n3 = L*row + (col+1)%L;
      n3 = blookup[(loc_grid[n3 >> 3] >> (n3 & 7)) & one];

      n4 = L*row + (col+L-1)%L;
      n4 = blookup[(loc_grid[n4 >> 3] >> (n4 & 7)) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n1 + n2 + n3 + n4 + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          ibyte = spin_index >> 3;
          index = spin_index & 7;

          //ibyte = spin_index / 8;
          //index = spin_index % 8; 
          loc_grid[ibyte] ^= one << index;
      } 
      
      
    } //end for


    // Copy local data back to device global memory
    state[idx] = localState;

    //for (row=0;row<L;row++){
    //  for (col=0;col<L;col++){
    for (spin_index=0;spin_index<L*L;spin_index++){
      d_ising_grids[L*L*idx + spin_index] = blookup[(loc_grid[spin_index >> 3] >> (spin_index & 7)) & one];
      //}
    }

    

  }

  return;

}

// Similar to mc_sweep_gpu_bitrep, but maps each thread in a block of 32 threads to a 
// fixed bit in a datatype of size 4 bytes for faster addressing.
__global__ void mc_sweep_gpu_bitmap32(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  const int llookup[2] = {-1, 1};

  // Shared memory for storage of bits
  uint32_t *bit_grid = (uint32_t *)&shared_grid[0];
  uint32_t one  = 1U;
  uint32_t zero = 0U;

  // Location in global memory where grids for the current block are stored
  int *block_grid = &d_ising_grids[L*L*blockIdx.x*blockDim.x];

  // Populate from global memory, ensuring that uint32_t is only written to by a single thread.
  int ispin,spin,ibit;
  for (ispin=threadIdx.x;ispin<L*L;ispin+=blockDim.x){
    bit_grid[ispin] = zero; 
    for (ibit=0;ibit<blockDim.x;ibit++){
      spin = block_grid[ibit*L*L + ispin];
      if ( spin == 1 ) {
        bit_grid[ispin] ^= one << ibit;
      }
    }
  }

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];
 
    int N=L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;
    //float shrink = (1.0f - FLT_EPSILON);
    int imove, row, col, index, my_idx, n1, n2 , n3, n4;

    for (imove=0;imove<N*nsweeps;imove++){

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));
      row = my_idx/L;
      col = my_idx%L;
 
      spin = llookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      
      // find neighbours
      n1 = llookup[(bit_grid[L*((row+1)%L) + col] >> threadIdx.x) & one];
      n2 = llookup[(bit_grid[L*((row+L-1)%L) + col] >> threadIdx.x) & one];
      n3 = llookup[(bit_grid[L*row + (col+1)%L] >> threadIdx.x) & one];
      n4 = llookup[(bit_grid[L*row + (col+L-1)%L] >> threadIdx.x) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n1 + n2 + n3 + n4 + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          //bit_grid[my_idx] ^= one << threadIdx.x;
          atomicXor(&bit_grid[my_idx],one << threadIdx.x);

      } 
      
      
    } //end for

    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        my_idx = L*row + col;
        d_ising_grids[N*idx+my_idx] = llookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      }
    }

  }

  return;

}

__global__ void mc_sweep_gpu_bitmap64(const int L, curandState *state, const int ngrids, int *d_ising_grids, int *d_neighbour_list, const float beta, const float h, int nsweeps) {

  const int llookup[2] = {-1, 1};

  // Shared memory for storage of bits
  uint64_t *bit_grid = (uint64_t *)&shared_grid[0];
  uint64_t one  = 1U;
  uint64_t zero = 0U;

  // Location in global memory where grids for the current block are stored
  int *block_grid = &d_ising_grids[L*L*blockIdx.x*blockDim.x];

  // Populate from global memory, ensuring that uint64_t is only written to by a single thread.
  int ispin,spin,ibit;
  for (ispin=threadIdx.x;ispin<L*L;ispin+=blockDim.x){
    bit_grid[ispin] = zero; 
    for (ibit=0;ibit<blockDim.x;ibit++){
      spin = block_grid[ibit*L*L + ispin];
      if ( spin == 1 ) {
        bit_grid[ispin] ^= one << ibit;
      }
    }
  }

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    // local copy of RNG state for current threads 
    curandState localState = state[idx];
  
    int N=L*L;
    float shrink = (1.0f - FLT_EPSILON)*(float)N;
    //float shrink = (1.0f - FLT_EPSILON);
    int imove, row, col, index, my_idx, n1, n2 , n3, n4;

    for (imove=0;imove<N*nsweeps;imove++){

      my_idx = __float2int_rd(shrink*curand_uniform(&localState));
      row = my_idx/L;
      col = my_idx%L;
 
      spin = llookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      
      // find neighbours
      n1 = llookup[(bit_grid[L*((row+1)%L) + col] >> threadIdx.x) & one];
      n2 = llookup[(bit_grid[L*((row+L-1)%L) + col] >> threadIdx.x) & one];
      n3 = llookup[(bit_grid[L*row + (col+1)%L] >> threadIdx.x) & one];
      n4 = llookup[(bit_grid[L*row + (col+L-1)%L] >> threadIdx.x) & one];

      //n_sum = 4;
      index = 5*(spin+1) + n1 + n2 + n3 + n4 + 4;

      if (curand_uniform(&localState) < d_Pacc[index] ) {
          // accept - toggle bit
          bit_grid[my_idx] ^= one << threadIdx.x;
          //atomicXor(&bit_grid[my_idx],one << threadIdx.x);  // unsupported for uint64_t
      } 
      
      
    } //end for

    // Copy local data back to device global memory
    state[idx] = localState;

    for (row=0;row<L;row++){
      for (col=0;col<L;col++){
        my_idx = L*row + col;
        d_ising_grids[N*idx+my_idx] = llookup[(bit_grid[my_idx] >> threadIdx.x) & one];
      }
    }

  }

  return;

}

// compute magnetisation on the gpu
__global__ void compute_magnetisation_gpu(const int L, const int ngrids, int *d_ising_grids, float *d_magnetisation) {

  int idx = threadIdx.x+blockIdx.x*blockDim.x;

  if (idx < ngrids) {

    int *loc_grid = &d_ising_grids[idx*L*L]; // pointer to device global memory

    float m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += loc_grid[i]; }
    d_magnetisation[idx] = m/(float)(L*L);

  }

  return;

}

__global__ void compute_largest_cluster_gpu(const int L, const int ngrids, int *d_ising_grids, const int spin, int *d_work, float *lclus_size) {

    /* Memory allocated here needs 2*ngrids*L*L*sizeof(int) as the minimum size 
       set in a call to cudaDeviceSetLimit() */

    int d_idx = threadIdx.x+blockIdx.x*blockDim.x;  

    //int* visited = (int*)malloc(L * L * sizeof(int));
    //for (int i=0;i<L*L;++i) {visited[i] = 0;}
    int* visited = d_work + d_idx*2*L*L; // Use part of the allocated work space
    for (int i=0;i<L*L;++i) {visited[i] = 0;}

    int max_size = 0;

    // Queue for BFS: stores indices
    //int* queue = (int*)malloc(L * L * sizeof(int));
    int* queue = d_work + d_idx*2*L*L + L*L;

    int front, back;

    // Neighbor offsets: left, right, up, down
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};



    if (d_idx < ngrids) {

      int *grid = &d_ising_grids[d_idx*L*L]; // pointer to device global memory

      for (int y = 0; y < L; ++y) {
          for (int x = 0; x < L; ++x) {
              int idx = y * L + x;
              if (grid[idx] == spin && !visited[idx]) {
                  visited[idx] = 1;
                  front = back = 0;
                  queue[back++] = idx;
                  int size = 0;

                  while (front < back) {
                      int current = queue[front++];
                      size++;

                      int cx = current % L;
                      int cy = current / L;

                      for (int d = 0; d < 4; ++d) {
                          int nx = (cx + dx[d] + L) % L;
                          int ny = (cy + dy[d] + L) % L;
                          int nidx = ny * L + nx;

                          if (grid[nidx] == spin && !visited[nidx]) {
                              visited[nidx] = 1;
                              queue[back++] = nidx;
                          }
                      }
                  }

                  if (size > max_size) {
                      max_size = size;
                  }
              }
          }
      }

      lclus_size[d_idx] = (float)max_size;

    }
    
}



void mc_driver_gpu(mc_gpu_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, gpu_run_t gpu_state, GridOutputFunc outfunc){

    clock_t t1,t2;  // For measuring time taken
    int isweep;     // MC sweep loop counter
    int igrid;      // counter for loop over replicas

    // Unpack structs
    int L = grids.L;
    int ngrids = grids.ngrids;
    int *ising_grids = grids.ising_grids;
    int *d_ising_grids = grids.d_ising_grids;
    int *d_neighbour_list = grids.d_neighbour_list;
    
    int tot_nsweeps = samples.tot_nsweeps;
    int mag_output_int = samples.mag_output_int;
    int grid_output_int = samples.grid_output_int;

    int itask = calc.itask;
    char *cv = calc.cv;
    double dn_thr = calc.dn_thr;
    double up_thr = calc.up_thr;
    int ninputs = calc.ninputs;
    int initial_spin = calc.initial_spin;
    char *filename = calc.filename;


    curandState* d_state = gpu_state.d_state;
    int threadsPerBlock = gpu_state.threadsPerBlock;
    int gpu_method = gpu_state.gpu_method;

    // Number of grids per input grid
    if (ngrids % ninputs != 0) {
      fprintf(stderr,"Error: ngrids must be divisible by ninputs!\n");
      exit(EXIT_FAILURE);
    }
    int sub_ngrids = ngrids/ninputs;


    // Pointer to collective variable array
    float *colvar = NULL;

    // Host copy of magnetisation
    float *magnetisation = (float *)malloc(ngrids*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation host array!\n");
      exit(EXIT_FAILURE);
    }

    // Device copy of magnetisation
    float *d_magnetisation;
    gpuErrchk( cudaMalloc(&d_magnetisation,ngrids*sizeof(float)) );

    
    // Only compute largest cluster if it's the CV of interest
    float *d_lclus = NULL, *lclus_size = NULL;
    int *d_work = NULL;
    if (strcmp(cv, "largest_cluster") == 0) {

      // Host copy of largest cluster size
      lclus_size = (float *)malloc(ngrids*sizeof(float));
      if (lclus_size==NULL){
        fprintf(stderr,"Error allocating largest cluster size host array!\n");
        exit(EXIT_FAILURE);
      }
      // Device copy of largest cluster size
      gpuErrchk( cudaMalloc(&d_lclus,ngrids*sizeof(float)) );

      // If we're using the largest cluster size calculation we need to allow for a large enough
      // heap size on the GPU to allocate arrays needed for the BFS algorithm.
      cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2*ngrids*L*L*sizeof(int));

      // Allocate workspace for the cluster size calculation on the device
      gpuErrchk( cudaMalloc(&d_work,2*ngrids*L*L*sizeof(int)) );

      colvar = lclus_size; // Use largest cluster size as collective variable

    } else {
      colvar = magnetisation; // Use the magnetisation as collective variable
    }


    // Streams
    cudaStream_t stream1;
    gpuErrchk( cudaStreamCreate(&stream1) );

    cudaStream_t stream2;
    gpuErrchk( cudaStreamCreate(&stream2) );


    // Allocate threads to thread blocks                                                         
    int blocksPerGrid = ngrids/threadsPerBlock;
    if (ngrids%threadsPerBlock!=0) { blocksPerGrid += 1; }      
      

    // How many sweeps to run in each call
    int sweeps_per_call;
    sweeps_per_call = mag_output_int < grid_output_int ? mag_output_int : grid_output_int;

    // result - either fraction of nucleated trajectories at each snapshot (itask=0) or comittor(s) (itask=1)
    float *result;
    int result_size;
    if (itask==0) {
      result_size = tot_nsweeps/mag_output_int;
    } else if (itask==1) {
      result_size = ninputs;
    } else {
      fprintf(stderr,"Error: itask must be 0 or 1!\n");
      exit(EXIT_FAILURE);
    }
    result=(float *)malloc(result_size*sizeof(float));
    if (result==NULL) {
      fprintf(stderr,"Error allocating result array!\n");
      exit(EXIT_FAILURE);
    }

    // Initialise result as nucleated in case code finishes after all reach stable state
    if (itask==0) {
      for (igrid=0;igrid<result_size;igrid++){
        result[igrid] = 1.0;
      }
    }

    t1 = clock();  // Start Timer

    isweep = 0;
    while(isweep < tot_nsweeps){

      // Output grids to file
      if (isweep%grid_output_int==0){
        // Asynchronous - can happen while magnetisation is being computed in stream 2
        gpuErrchk( cudaMemcpyAsync(ising_grids,d_ising_grids,L*L*ngrids*sizeof(int),cudaMemcpyDeviceToHost,stream1) );
      }

      // Can compute collective variables which magnetisation while grids are copying
      if (isweep%mag_output_int==0){

        compute_magnetisation_gpu<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(L, ngrids, d_ising_grids, d_magnetisation);    
        gpuErrchk( cudaMemcpyAsync(magnetisation,d_magnetisation,ngrids*sizeof(float),cudaMemcpyDeviceToHost, stream2) );

        if (strcmp(cv, "largest_cluster") == 0) {
          compute_largest_cluster_gpu<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(L, ngrids, d_ising_grids, -1*initial_spin, d_work, d_lclus); // spin=1
          gpuErrchk( cudaMemcpyAsync(lclus_size,d_lclus,ngrids*sizeof(float),cudaMemcpyDeviceToHost, stream2) );
        }

      } 

      // MC Sweep - GPU
      gpuErrchk( cudaStreamSynchronize(stream1) ); // Make sure copy completed before making changes

      if (gpu_method==0){
        mc_sweep_gpu<<<blocksPerGrid,threadsPerBlock,0,stream1>>>(L,d_state,ngrids,d_ising_grids,d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
      } else if (gpu_method==1){
          size_t shmem_size = ceil(L*L/8)*threadsPerBlock*sizeof(uint8_t); // number of bytes needed to store grid as bits
          cudaFuncSetAttribute(mc_sweep_gpu_bitrep, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
          cudaFuncSetAttribute(mc_sweep_gpu_bitrep, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
          mc_sweep_gpu_bitrep<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
          gpuErrchk( cudaGetLastError());
      } else if (gpu_method==2){
          size_t shmem_size = ceil(L*L/8)*threadsPerBlock*sizeof(uint8_t); // number of bytes needed to store grid as bits
          if (threadsPerBlock==32){
            cudaFuncSetAttribute(mc_sweep_gpu_bitmap32, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            cudaFuncSetAttribute(mc_sweep_gpu_bitmap32, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            mc_sweep_gpu_bitmap32<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
            gpuErrchk( cudaGetLastError());
          } else if (threadsPerBlock==64){
            cudaFuncSetAttribute(mc_sweep_gpu_bitmap64, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            cudaFuncSetAttribute(mc_sweep_gpu_bitmap64, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
            mc_sweep_gpu_bitmap64<<<blocksPerGrid,threadsPerBlock,shmem_size,stream1>>>(L,d_state,ngrids,d_ising_grids, d_neighbour_list, (float)beta,(float)h,sweeps_per_call);
            gpuErrchk( cudaGetLastError());
          } else {
            fprintf(stderr, "Invalid threadsPerBlock for gpu_method=2\n");
            exit(EXIT_FAILURE);
          } 
      } else {
        fprintf(stderr, "Unknown gpu_method in ising.cu\n");
        exit(EXIT_FAILURE);
      }
      
      // Writing of the grids can be happening on the host while the device runs the mc_sweep kernel
      if (isweep%grid_output_int==0 || isweep==tot_nsweeps-1){
        outfunc(L, ngrids, ising_grids, isweep, magnetisation, lclus_size, cv, dn_thr, up_thr, filename);  
      }

      // Write and report cv - can also be happening while the device runs the mc_sweep kernel
      if (isweep%mag_output_int==0 || isweep==tot_nsweeps-1){
        gpuErrchk( cudaStreamSynchronize(stream2) );  // Wait for copy to complete
        //for (igrid=0;igrid<ngrids;igrid++){
        //  printf("    %4d     %10d      %8.6f\n",igrid, isweep, magnetisation[igrid]);
        //}
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<ngrids;igrid++){
            if ( colvar[igrid] > up_thr ) nnuc++;
          }
          result[isweep/mag_output_int] = (float)((double)nnuc/(double)ngrids);
#ifndef PYTHON
          fprintf(stdout, "%10d  %12.6f\n",isweep, (double)nnuc/(double)ngrids);
          fflush(stdout);
#endif 
#ifdef PYTHON
          PySys_WriteStdout("\r Sweep : %10d, Reached cv = %6.2f : %4d , Unresolved : %4d",
            isweep, nnuc, up_thr, ngrids-nnuc );
#endif
          if (nnuc==ngrids) break; // Stop if everyone has nucleated
        } else if ( itask == 1 ){

            // Statistics on fate of trajectories
            int nA=0, nB=0;
            for (igrid=0;igrid<ngrids;igrid++){
              if (grid_fate[igrid]==0 ) {
                nA++;
              } else if (grid_fate[igrid]==1 ) {
                nB++;
              } else {
                if ( colvar[igrid] > up_thr ){
                  grid_fate[igrid] = 1;
                  nB++;
                } else if (colvar[igrid] < dn_thr ){
                  grid_fate[igrid] = 0;
                  nA++;
                }
              } // fate
            } //grids

            // Monitor progress
#ifndef PYTHON
            printf("\r Sweep : %10d, Reached cv = %6.2f : %4d , Reached cv = %6.2f : %4d , Unresolved : %4d",
            isweep, dn_thr, nA, up_thr, nB, ngrids-nA-nB);
            //printf("\r colvar : %10d",colvar[0]);
            fflush(stdout);
#endif
#ifdef PYTHON
            PySys_WriteStdout("\r Sweep : %10d, Reached cv = %6.2f : %4d , Reached cv = %6.2f : %4d , Unresolved : %4d    ",
            isweep, dn_thr, nA, up_thr, nB, ngrids-nA-nB );
            //PySys_WriteStdout("\r colvar : %10f",colvar[0]);
#endif
            if (nA + nB == ngrids) break; // All fates resolved
        
        } // task 
      }

      // Increment isweep
      isweep += sweeps_per_call;

      // Make sure all kernels updating the grids are finished before starting magnetisation calc
      gpuErrchk( cudaStreamSynchronize(stream1) );
      gpuErrchk( cudaPeekAtLastError() );

    }

    // Ensure all threads finished before stopping timer
    gpuErrchk( cudaDeviceSynchronize() );

    t2 = clock();

#ifndef PYTHON
    fprintf(stdout, "\n# Time taken on GPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);
    if (itask==1) { fprintf(stdout, "pB estimate : %10.6f\n", result); }
    fflush(stdout);
#endif
#ifdef PYTHON
    PySys_WriteStdout("\n");
#endif


    // Destroy streams
    gpuErrchk( cudaStreamDestroy(stream1) );
    gpuErrchk( cudaStreamDestroy(stream2) );


    // Free magnetisation arrays
    free(magnetisation);
    if (lclus_size) free(lclus_size);
    gpuErrchk( cudaFree(d_magnetisation) );
    
    if (d_lclus) gpuErrchk( cudaFree(d_lclus) );
    if (d_work) gpuErrchk( cudaFree(d_work) );

    if (itask==0) { // Fraction of nucleated grids
      for (int i = 0; i < result_size; i++) {
        calc.result[i] = result[i];
      }
    } else if (itask==1) { // Compute the committor(s)
      int ii;
      for (ii=0;ii<ninputs;ii++) {
        int nB = 0;
        int nF = 0;
        for (int jj=0;jj<sub_ngrids;jj++) {
          if (grid_fate[ii*sub_ngrids+jj] > -1) {
            nB += grid_fate[ii*sub_ngrids+jj];
          }
          else {
            nF += 1;
          }
        }
        calc.result[ii] = (float)nB/(float)(sub_ngrids-nF); // Copy result to output array
      }
    }
      
    if (result) free(result);

}
