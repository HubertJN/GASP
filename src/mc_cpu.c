#include <stdio.h>
#include <time.h>
#include "mc_cpu.h"
#include "io.h"

#ifdef PYTHON
#include <Python.h>
#endif

double Pacc[20];         // Cache of acceptance probabilities


// populate acceptance probabilities
void preComputeProbs_cpu(double beta, double h) {

  // Pre-compute acceptance probabilities for all possibilities
  // s  : nsum               : index 5*(s+1) + nsum + 4
  // +1 : -4 = -1 -1 -1 -1   : 10
  // +1 : -2 = -1 -1 -1 +1   : 12
  // +1 :  0 = -1 -1 +1 +1   : 14
  // +1 : +2 = -1 +1 +1 +1   : 16
  // +1 : +4 = +1 +1 +1 +1   : 18
  // -1 : -4 = -1 -1 -1 -1   : 0
  // -1 : -2 = -1 -1 -1 +1   : 2
  // -1 :  0 = -1 -1 +1 +1   : 4
  // -1 : +2 = -1 +1 +1 +1   : 6
  // -1 : +4 = +1 +1 +1 +1   : 8

  int s, nsum, index;
  for (s=-1;s<2;s=s+2){
    for (nsum=-4;nsum<5;nsum=nsum+2){
      index = 5*(s+1) + nsum + 4;
      Pacc[index] = 2.0*(double)s*((double)nsum+h);
      Pacc[index] = exp(-beta*Pacc[index]);
    }
  }

}     

// sweep on the cpu
void mc_sweep_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps) {

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

  int imove, row, col;

  for (imove=0;imove<L*L*nsweeps;imove++){

    // pick random spin
    row = floor(L*genrand_real3());  // RNG cannot generate 1.0 so safe
    col = floor(L*genrand_real3()); 

    // find neighbours
    int my_idx = L*row+col;
    int up_idx = L*((row+L-1)%L) + col;  
    int dn_idx = L*((row+1)%L)   + col;   
    int rt_idx = L*row + (col+1)%L;
    int lt_idx = L*row + (col+L-1)%L;
    
    // energy before flip
    int n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_old = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    int index = 5*(loc_grid[my_idx]+1) + n_sum + 4;

    // flip
    loc_grid[my_idx] = -1*loc_grid[my_idx];

    // energy after flip
    //n_sum = loc_grid[up_idx] + loc_grid[dn_idx] + loc_grid[lt_idx] + loc_grid[rt_idx]; 
    //double energy_new = -1.0 * (double)loc_grid[my_idx] * ( (double)n_sum + h );

    //double delta_energy = energy_new - energy_old;
    //double prob = exp(-beta*delta_energy);

    double xi = genrand_real3();
    //if (xi < prob) {
    if (xi < Pacc[index] ) {
      // accept
      //fprintf(stderr,"Accepted a move\n");
    } else {
      loc_grid[my_idx] = -1*loc_grid[my_idx]; // reject
    }


  } // end for

}

void compute_magnetisation_cpu(int L, int *ising_grids, int grid_index, float *magnetisation){

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

    double m = 0.0f;

    int i;
    for (i=0;i<L*L;i++) { m += (double)loc_grid[i]; }
    magnetisation[grid_index] = (float)(m/(double)(L*L));

  return;

}

void mc_driver_cpu(mc_grids_t grids, double beta, double h, int* grid_fate, mc_sampler_t samples, mc_function_t calc, GridOutputFunc outfunc){

    clock_t t1,t2;  // For measuring time taken
    int isweep;     // MC sweep loop counter
    int igrid;      // counter for loop over replicas

    // Unpack structs
    int L = grids.L;
    int ngrids = grids.ngrids;
    int *ising_grids = grids.ising_grids;

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

    // Number of grids per input grid
    if (ngrids % ninputs != 0) {
      fprintf(stderr,"Error: ngrids must be divisible by ninputs!\n");
      exit(EXIT_FAILURE);
    }
    int sub_ngrids = ngrids/ninputs;

    float *colvar; // Collective variable

    // How many sweeps to run in each call to mc_sweeps_cpu
    int sweeps_per_call;
    sweeps_per_call = mag_output_int < grid_output_int ? mag_output_int : grid_output_int;
    
    // Magnetisation of each grid - cheap to compute so always allocated
    float *magnetisation = (float *)malloc(ngrids*sizeof(float));
    if (magnetisation==NULL){
      fprintf(stderr,"Error allocating magnetisation array!\n");
      exit(EXIT_FAILURE);
    }

    // Largest cluster size for each grid only if we need it
    float *lclus = NULL;
    if (strcmp(cv, "largest_cluster") == 0) {
      lclus = (float *)malloc(ngrids*sizeof(float));
      if (lclus==NULL){
        fprintf(stderr,"Error allocating largest cluster size array!\n");
        exit(EXIT_FAILURE);
      }
      colvar = lclus; // Use largest cluster size as collective variable
    } else {
      colvar = magnetisation; // Use the magnetisation as collective variable
    }



    // result - either fraction of nucleated trajectories (itask=0) or comittor(s) (itask=1)
    float *result;
    int *nucleated = NULL; 
    int result_size;
    if (itask==0) {
      result_size = tot_nsweeps/mag_output_int;
      nucleated = (int *)malloc(ngrids*sizeof(int));
      if (nucleated==NULL) {
        fprintf(stderr,"Error allocating nucleated array!\n");
        exit(EXIT_FAILURE);
      }
      for (igrid=0;igrid<ngrids;igrid++){
        nucleated[igrid] = -1; // Initialise as not nucleated
      }
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

    t1 = clock();  // Start timer

    isweep = 0;
    while (isweep < tot_nsweeps){

      // Report collective variables
      if (isweep%mag_output_int==0){
        for (igrid=0;igrid<ngrids;igrid++){

          compute_magnetisation_cpu(L, ising_grids, igrid, magnetisation);
          if ( strcmp(cv, "largest_cluster") == 0 ) {
            compute_largest_cluster_cpu(L, ising_grids, igrid, -1*initial_spin, lclus);
          }

        }
        if ( itask == 0 ) { // Report how many samples have nucleated.
          int nnuc = 0;
          for (igrid=0;igrid<ngrids;igrid++){
            if (( colvar[igrid] > up_thr) && (nucleated[igrid] == -1 )) {
              nucleated[igrid] = 1; // Mark as nucleated
            }
            if (nucleated[igrid] == 1) {
              nnuc++; 
            }
          }
#ifndef PYTHON
          fprintf(stdout, "%10d  %12.6f\n",isweep, (double)nnuc/(double)ngrids);
#endif
#ifdef PYTHON
          PySys_WriteStdout("\r Sweep : %10d, Reached cv = %6.2f : %4d , Unresolved : %4d",
            isweep, nnuc, up_thr, ngrids-nnuc );
#endif
          result[isweep/mag_output_int] = (float)((double)nnuc/(double)ngrids);
	  if (nnuc==ngrids) break; // Stop if everyone has nucleated
	  
        } else if ( itask == 1 ){

          // Statistics on fate of trajectories
          int nA=0, nB=0;
          for (igrid=0;igrid<ngrids;igrid++){
            //printf("grid_fate[%d] = %d\n",igrid, grid_fate[igrid]);
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
          fprintf(stdout, "\r Sweep : %10d, Reached cv = %6.2f : %4d , Reached cv = %6.2f : %4d , Unresolved : %4d",
		 isweep, dn_thr, nA, up_thr, nB, ngrids-nA-nB);
          fflush(stdout);
#endif
#ifdef PYTHON
            PySys_WriteStdout("\r Sweep : %10d, Reached cv = %6.2f : %4d , Reached cv = %6.2f : %4d , Unresolved : %4d",
            isweep, dn_thr, nA, up_thr, nB, ngrids-nA-nB );
            //PySys_WriteStdout("\r colvar : %10f",colvar[0]);
#endif


          if (nA + nB == ngrids) break; // all fates resolved
        } // task
      } 

      // Output grids to file
      if (isweep%grid_output_int==0){
        outfunc(L, ngrids, ising_grids, isweep, magnetisation, lclus, cv, dn_thr, up_thr, filename);  
      }


      // MC Sweep - CPU
      for (igrid=0;igrid<ngrids;igrid++) {
        mc_sweep_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
      }
      isweep += sweeps_per_call;

    }
      
    t2 = clock();  // Stop Timer

#ifndef PYTHON
    fprintf(stdout, "\n# Time taken on CPU = %f seconds\n",(double)(t2-t1)/(double)CLOCKS_PER_SEC);

    if (itask==1) { printf("pB estimate : %10.6f\n",result); }; 
#endif
#ifdef PYTHON
    PySys_WriteStdout("\n");
#endif

    // Release memory
    free(magnetisation);  
    if (lclus) free(lclus);

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
    if (nucleated) free(nucleated);


}

void compute_largest_cluster_cpu(int L, int* ising_grids, const int grid_index, int spin, float *lclus_size){

    int* visited = (int*)calloc(L * L, sizeof(int));
    int max_size = 0;

    // Queue for BFS: stores indices
    int* queue = (int*)malloc(L * L * sizeof(int));
    int front, back;

    // Neighbor offsets: left, right, up, down
    int dx[4] = {-1, 1, 0, 0};
    int dy[4] = {0, 0, -1, 1};

    // Part of ising_grids array to work on
    int *grid = &ising_grids[grid_index*L*L];

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

    free(visited);
    free(queue);
    lclus_size[grid_index] = (float)max_size;
}
