#include <stdio.h>
#include <time.h>
#include "mc_cpu.h"
#include "io.h"

#ifdef PYTHON
#include <Python.h>
#endif

double Pacc[20];         // Cache of acceptance probabilities for spin flip moves
double PaccKaw_2NN[100]; // Cache of acceptance probabilities for Kawasaki moves


// Compute the index within PaccKaw_2NN for a given move. The index is determined 
// by the spins of the two sites being swapped (Si, Sj) and the sum of their 4 nearest
// neighbours (Ni, Nj). Each of these can take on a limited number of values, so we can
// pre-compute the acceptance probabilities for all combinations and store them in an
// array for fast lookup during the simulation.
static inline int kawasaki_index_4NN(int si, int sj, int n_i, int n_j) {
  int Si = (si + 1) / 2;    // 0 or 1
  int Sj = (sj + 1) / 2;    // 0 or 1
  int Ni = (n_i + 4) / 2;   // 0..4
  int Nj = (n_j + 4) / 2;   // 0..4
  return (((Si * 2 + Sj) * 5 + Ni) * 5 + Nj); // 0..99
}


// populate acceptance probabilities - Kawasaki moves.
// NOTE: 'h' is ignored for exchange with uniform field (it cancels).
void preComputeProbsKawasaki_4NN_cpu(double beta, double h_ignored) {

  int si_vals[2] = {-1, +1};
  int sj_vals[2] = {-1, +1};
  int n_vals[5]  = {-4, -2, 0,  2,  4};

  for (int a = 0; a < 2; ++a) {
    int si = si_vals[a];
    for (int b = 0; b < 2; ++b) {
      int sj = sj_vals[b];

      for (int u = 0; u < 5; ++u) {
        int n_i = n_vals[u];
        for (int v = 0; v < 5; ++v) {
          int n_j = n_vals[v];

          int idx = kawasaki_index_4NN(si, sj, n_i, n_j);

          double dE;
          if (si == sj) {
            // Swapping identical spins leaves configuration unchanged.
            dE = 0.0;
          } else {
            // Use 3-NN sums derived from 4-NN sums:
            int Sigma_i_excl = n_i - sj; // sum of the 3 neighbors of i excluding j
            int Sigma_j_excl = n_j - si; // sum of the 3 neighbors of j excluding i
            dE = 2.0 * (si * Sigma_i_excl + sj * Sigma_j_excl); // J = 1
          }

          double w = exp(-beta * dE);
          // If you want pure Metropolis prob, clamp to 1.0:
          // if (w > 1.0) w = 1.0;
          PaccKaw_2NN[idx] = w;
          //PySys_WriteStdout("Kawasaki move: index=%d, Si=%d, Sj=%d, Ni=%d, Nj=%d => dE=%.2f, Pacc=%.4f\n", 
          //        idx, si, sj, n_i, n_j, dE, w);
        }
      }
    }
  }
}





// populate acceptance probabilities - spin flip moves
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
void mc_sweep_kawasaki_cpu(int L, int *ising_grids, int grid_index, double beta, double h, int nsweeps) {

  // Pointer to the current Ising grid
  int *loc_grid = &ising_grids[grid_index*L*L];

  int imove, row, col;

  // count number of up spins
  int n_up = 0;
  for (int i = 0; i < L*L; i++) {
    if (loc_grid[i] == 1) n_up++;
  }

  for (imove=0;imove<n_up*nsweeps;imove++){

    // pick random spin
    //row = floor(L*genrand_real3());  // RNG cannot generate 1.0 so safe
    //col = floor(L*genrand_real3()); 

    // pick a random spin up site
    do {
      row = floor(L*genrand_real3());
      col = floor(L*genrand_real3());
    } while (loc_grid[L*row+col] != 1); // Keep picking until we find a spin up site

    // find neighbours
    int my_idx = L*row+col;
    int my_up_idx = L*((row+L-1)%L) + col;  
    int my_dn_idx = L*((row+1)%L)   + col;   
    int my_rt_idx = L*row + (col+1)%L;
    int my_lt_idx = L*row + (col+L-1)%L;
    
    // pick random neighbour to swap with
    int n_idx;
    int n_row, n_col;
    int direction = floor(4*genrand_real3());
    if (direction == 0) { // up
      n_row = (row + L - 1) % L;
      n_col = col;
      n_idx = my_up_idx;
    } else if (direction == 1) { // down
      n_row = (row + 1) % L;
      n_col = col;
      n_idx = my_dn_idx;
    } else if (direction == 2) { // right
      n_row = row;
      n_col = (col + 1) % L;
      n_idx = my_rt_idx;
    } else { // left
      n_row = row;
      n_col = (col + L - 1) % L;
      n_idx = my_lt_idx;
    }

    // find neighbours of n_idx
    int n_up_idx = L*((n_row+L-1)%L) + n_col;  
    int n_dn_idx = L*((n_row+1)%L)   + n_col;   
    int n_rt_idx = L*n_row + (n_col+1)%L;
    int n_lt_idx = L*n_row + (n_col+L-1)%L;

    // compute index for acceptance probability lookup
    int si = loc_grid[my_idx];
    int sj = loc_grid[n_idx];
    int n_i = loc_grid[my_up_idx] + loc_grid[my_dn_idx] + loc_grid[my_lt_idx] + loc_grid[my_rt_idx];
    int n_j = loc_grid[n_up_idx] + loc_grid[n_dn_idx] + loc_grid[n_lt_idx] + loc_grid[n_rt_idx];
    int index = kawasaki_index_4NN(si, sj, n_i, n_j);

    // perform swap    int temp = loc_grid[my_idx];
    int temp = loc_grid[my_idx];
    loc_grid[my_idx] = loc_grid[n_idx];
    loc_grid[n_idx] = temp;
    
    double xi = genrand_real3();
    //if (xi < prob) {
    if (xi < PaccKaw_2NN[index] ) {
      // accept
      //printf("Accepted a move\n");
    } else {
      // reject - swap back
      temp = loc_grid[my_idx];
      loc_grid[my_idx] = loc_grid[n_idx];
      loc_grid[n_idx] = temp;
    }


  } // end for

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
    char *dynamics = calc.dynamics;
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
            if ( colvar[igrid] > up_thr ) nnuc++;
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
      if (strcmp(dynamics, "kawasaki") == 0) {
        for (igrid=0;igrid<ngrids;igrid++) {
          mc_sweep_kawasaki_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
        }
      } else { // default to spin flip
          for (igrid=0;igrid<ngrids;igrid++) {
            mc_sweep_cpu(L, ising_grids, igrid, beta, h, sweeps_per_call);
          }
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
