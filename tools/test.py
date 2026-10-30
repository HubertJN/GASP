import gasp

L = 64 
ngrids = 4 * gasp.gpu_nsms * 32
nsweeps = 500

beta = 0.53
h = 0.064

grid_output_int = 100
mag_output_int = 100

frac = gasp.run_nucleation_swarm(
    L,
    ngrids,
    nsweeps,
    beta,
    h,
    grid_output_int=grid_output_int,
    mag_output_int=mag_output_int,
    cv="largest_cluster",
    up_threshold=400.5,
    dn_threshold=9.5,
    keep_grids=False,
    )
