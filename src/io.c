#include "io.h"
#include <hdf5.h>
#include <math.h>

void read_input_grid(int L, int ngrids, int *ising_grids){

    // converts [0,1] to [-1,1]
    const int blookup[2] = {-1, 1};

    // Set filename
    char filename[14];
    sprintf(filename, "gridinput.bin");

    uint32_t one = 1U;   

    // open file
    FILE *ptr = fopen(filename, "rb");
    if (ptr==NULL){
        fprintf(stderr, "Error opening %s for input!\n", filename);
        exit(EXIT_FAILURE);
    }

    // read header specifying size of grid
    int Lcheck;
    fread(&Lcheck, sizeof(int), 1, ptr);
    if (Lcheck!=L) {
        fprintf(stderr, "Error - size of grid in input file does not match L!\n");
        exit(EXIT_FAILURE);
    }

    // Allocate space to read a single grid as bits
    int nbytes = L*L/8;
    if ( (L*L)%8 !=0 ) { nbytes++; }
    char *bitgrid = (char *)malloc(nbytes);
    if (bitgrid==NULL){
        fprintf(stderr,"Error allocating input buffer!");
        exit(EXIT_FAILURE);
    }

    // Read the grid
    fread(bitgrid, sizeof(char), nbytes, ptr);  

    // Loop over grid points
    int ibit=0, ibyte=0;
    int isite=0, igrid;

    //printf("nbytes = %d\n",nbytes);
    for (ibyte=0;ibyte<nbytes;ibyte++){
        for (ibit=0;ibit<8;ibit++){
            //printf(" %2d ",blookup[(bitgrid[ibyte] >> ibit) & one]);
            // Read into every copy of the grid
            for (igrid=0;igrid<ngrids;igrid++){
                ising_grids[L*L*igrid+isite] = blookup[(bitgrid[ibyte] >> ibit) & one];
            }
            isite++;
            //if (isite%L==0) {printf("\n");}
        }
        if (isite>L*L) break;
    }

    free(bitgrid);  // free input buffer
    fclose(ptr);    // close input file

    fprintf(stderr, "Read initial configuration of all grids from gridinput.bin\n");

}

int write_ising_grids(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr, char *filename){

    // Set filename
    //char filename[15];
    //sprintf(filename, "gridstates.bin");
    //printf("%s\n",filename);

    // open file
    FILE *ptr = fopen(filename,"ab");
    if (ptr==NULL){
        FILE *ptr = fopen(filename,"wb"); // open for write if not available for append 
        if (ptr==NULL){
            fprintf(stderr,"Error opening %s for write!\n",filename);
            exit(EXIT_FAILURE);
        }   
    }

    // file header - size of grid, number of grids and current sweep
    fwrite(&L,sizeof(int),1,ptr);
    fwrite(&ngrids,sizeof(int),1,ptr);
    fwrite(&isweep,sizeof(int),1,ptr);

    // pack everything into as few bits as possible
    int nbytes = L*L*ngrids/8;
    if ( (L*L*ngrids)%8 !=0 ) { nbytes++; }
    char *bitgrids = (char *)malloc(nbytes);
    if (bitgrids==NULL){
        fprintf(stderr,"Error allocating output buffer!");
        exit(EXIT_FAILURE);
    }

    // Set zero 
    memset(bitgrids, 0U, nbytes);
    
    uint8_t one = 1;

    int ibit=0, ibyte=0, iint;
    for (iint=0;iint<L*L*ngrids;iint++){ //loop over grid x squares per grid

        if ( ising_grids[iint] == 1 ) {
            bitgrids[ibyte] |= one << ibit;
            //printf("Set bit %d of byte %d\n", ibit, ibyte);
        }

        ibit++;
        if (ibit==8) {
            ibit=0;
            ibyte++;
        }

    }

    // write to file
    fwrite(bitgrids,sizeof(char),nbytes,ptr);

    // Release memory
    free(bitgrids);

    // close file
    fclose(ptr);

    return 0; //Success
    
}

int create_ising_grids_hdf5(int L, int ngrids, int tot_nsweeps, double h, double beta, int itask, char* filename) {
    
    /* Create HDF5 file */
    hid_t file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: failed to create HDF5 file '%s'\n", filename);
        return -1;
    }

    /* Create scalar dataspace for headers */
    hid_t sid = H5Screate(H5S_SCALAR);
    if (sid < 0) {
        fprintf(stderr, "Error: failed to create HDF5 scalar dataspace\n");
        H5Fclose(file_id);
        return -1;
    }

    /* Write integer scalars */
    int int_vals[3] = {L, ngrids, tot_nsweeps};
    const char *int_names[3] = {"L", "ngrids", "tot_nsweeps"};
    for (int i = 0; i < 3; ++i) {
        hid_t did = H5Dcreate2(file_id, int_names[i], H5T_NATIVE_INT, sid,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (did < 0) { fprintf(stderr, "Error: failed to create dataset '%s'\n", int_names[i]); continue; }
        H5Dwrite(did, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &int_vals[i]);
        H5Dclose(did);
    }

    /* Write double scalars */
    double dbl_vals[2] = {h, beta};
    const char *dbl_names[2] = {"h", "beta"};
    for (int i = 0; i < 2; ++i) {
        hid_t did = H5Dcreate2(file_id, dbl_names[i], H5T_NATIVE_DOUBLE, sid,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (did < 0) { fprintf(stderr, "Error: failed to create dataset '%s'\n", dbl_names[i]); continue; }
        H5Dwrite(did, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &dbl_vals[i]);
        H5Dclose(did);
    }

    /* Create total_saved_grids header initialized to 0 */
    hid_t did_tot = H5Dcreate2(file_id, "total_saved_grids", H5T_NATIVE_HSIZE, sid,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (did_tot >= 0) {
        hsize_t total = 0;
        H5Dwrite(did_tot, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &total);
        H5Dclose(did_tot);
    } else {
        fprintf(stderr, "Warning: failed to create dataset 'total_saved_grids'\n");
    }

    /* Done with scalar dataspace */
    H5Sclose(sid);

    /* Compute bytes per packed grid */
    size_t nbits = (size_t)L * (size_t)L;
    size_t nbytes = (nbits + 7) / 8;

    /* Create chunked datasets for grids and attrs (unlimited first dimension) */
    hsize_t dims[2] = {0, nbytes};
    hsize_t maxdims[2] = {H5S_UNLIMITED, nbytes};
    hid_t space = H5Screate_simple(2, dims, maxdims);
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_g[2] = {1, nbytes};
    H5Pset_chunk(dcpl, 2, chunk_g);
    hid_t d_grids = H5Dcreate2(file_id, "grids", H5T_NATIVE_UCHAR, space,
                               H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dclose(d_grids);
    H5Sclose(space);
    H5Pclose(dcpl);

    /* Create attrs dataset: 4 doubles per row */
    hsize_t adims[2] = {0, 4};
    hsize_t amax[2] = {H5S_UNLIMITED, 4};
    space = H5Screate_simple(2, adims, amax);
    dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t achunk[2] = {1024, 4};
    H5Pset_chunk(dcpl, 2, achunk);
    hid_t d_attrs = H5Dcreate2(file_id, "attrs", H5T_NATIVE_DOUBLE, space,
                               H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Dclose(d_attrs);
    H5Sclose(space);
    H5Pclose(dcpl);

    /* Close file */
    H5Fclose(file_id);

    return 0;
}

int write_ising_grids_hdf5(int L, int ngrids, int *ising_grids, int isweep, float *magnetisation, float *lclus_size, char *cv, double dn_thr, double up_thr, char* filename) {
    /* Setup data output filter based on thresholds*/
    float *filter = NULL;
    if (strcmp(cv, "magnetisation") == 0) {
        filter = magnetisation;
    } else if (strcmp(cv, "largest_cluster") == 0) {
        filter = lclus_size;
    } else {
        fprintf(stderr, "Error: invalid cv '%s'\n", cv);
        exit(1);
    }

    //const char *filename = "gridstates.hdf5";
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file_id < 0) {
        fprintf(stderr, "Error: cannot open '%s'\n", filename);
        return -1;
    }

    /* Compute bytes per (packed) grid */
    size_t nbits = (size_t)L * (size_t)L;
    size_t nbytes = (nbits + 7) / 8;

    /* Dataset creation property list (chunked) */
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunk_g[2] = { 1, (hsize_t)nbytes };
    H5Pset_chunk(dcpl, 2, chunk_g);

    /* Open datasets directly (assume they already exist) */
    hid_t d_grids = H5Dopen2(file_id, "grids", H5P_DEFAULT);
    hid_t d_attrs = H5Dopen2(file_id, "attrs", H5P_DEFAULT);
    hid_t d_tot   = H5Dopen2(file_id, "total_saved_grids", H5P_DEFAULT);

    if (d_grids < 0 || d_attrs < 0 || d_tot < 0) {
        fprintf(stderr, "Error: failed to open one or more datasets\n");
        if (d_grids >= 0) H5Dclose(d_grids);
        if (d_attrs >= 0) H5Dclose(d_attrs);
        if (d_tot >= 0) H5Dclose(d_tot);
        H5Pclose(dcpl);
        H5Fclose(file_id);
        return -1;
    }

    /* Get current sizes */
    hid_t gs = H5Dget_space(d_grids);
    hsize_t gdims[2];
    H5Sget_simple_extent_dims(gs, gdims, NULL);
    H5Sclose(gs);
    hsize_t start_idx = gdims[0];

    hid_t as = H5Dget_space(d_attrs);
    hsize_t adims_cur[2];
    H5Sget_simple_extent_dims(as, adims_cur, NULL);
    H5Sclose(as);

    /* Count how many grids pass filter */
    int nsave = 0;
    for (int g = 0; g < ngrids; g++) {
        //printf("%f %f %f \n", dn_thr, filter[g], up_thr);
        if (filter[g] >= dn_thr && filter[g] <= up_thr)
            nsave++;
    }

    if (nsave == 0) {
        //printf("No grids passed the filter.\n");
        H5Dclose(d_grids);
        H5Dclose(d_attrs);
        H5Dclose(d_tot);
        H5Pclose(dcpl);
        H5Fclose(file_id);
        return 0;
    } else {
        //printf("%d grids passed the filter and will be saved.\n", nsave);
    }


    /* Extend datasets */
    hsize_t new_gdims[2] = { start_idx + (hsize_t)nsave, (hsize_t)nbytes };
    H5Dset_extent(d_grids, new_gdims);

    hsize_t new_adims[2] = { adims_cur[0] + (hsize_t)nsave, 4 };
    H5Dset_extent(d_attrs, new_adims);

    /* Allocate memory buffers */
    unsigned char *buf = (unsigned char *)malloc(nbytes);
    if (!buf) {
        fprintf(stderr, "Error: failed to allocate memory\n");
        exit(1);
    }

    double attrrow[4];
    hsize_t write_idx = 0;

    /* Write grids and attributes */
    for (int g = 0; g < ngrids; ++g) {
        if (filter[g] < dn_thr || filter[g] > up_thr) continue;  // skip outside range

        memset(buf, 0, nbytes);
        int *grid_ptr = &ising_grids[g * L * L];
        size_t ibit = 0, ibyte = 0;
        for (size_t i = 0; i < nbits; ++i) {
            if (grid_ptr[i] == 1) buf[ibyte] |= (unsigned char)(1U << ibit);
            ibit++; if (ibit == 8) { ibit = 0; ibyte++; }
        }

        /* Write grids hyperslab */
        hid_t filespace = H5Dget_space(d_grids);
        hsize_t start[2] = { start_idx + write_idx, 0 };
        hsize_t count[2] = { 1, (hsize_t)nbytes };
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, NULL, count, NULL);
        hsize_t mdims[2] = { 1, (hsize_t)nbytes };
        hid_t mspace = H5Screate_simple(2, mdims, NULL);
        H5Dwrite(d_grids, H5T_NATIVE_UCHAR, mspace, filespace, H5P_DEFAULT, buf);
        H5Sclose(mspace);
        H5Sclose(filespace);

        /* Write attrs hyperslab */
        attrrow[0] = (magnetisation ? (double)magnetisation[g] : NAN);
        attrrow[1] = (lclus_size ? (double)lclus_size[g] : NAN);
        attrrow[2] = NAN; /* committor placeholder */
        attrrow[3] = NAN; /* committor_error placeholder */

        hid_t afilespace = H5Dget_space(d_attrs);
        hsize_t astart[2] = { adims_cur[0] + write_idx, 0 };
        hsize_t acount[2] = { 1, 4 };
        H5Sselect_hyperslab(afilespace, H5S_SELECT_SET, astart, NULL, acount, NULL);
        hsize_t amdims[2] = { 1, 4 };
        hid_t amspace = H5Screate_simple(2, amdims, NULL);
        H5Dwrite(d_attrs, H5T_NATIVE_DOUBLE, amspace, afilespace, H5P_DEFAULT, attrrow);
        H5Sclose(amspace);
        H5Sclose(afilespace);

        write_idx++;
    }

    /* Update total_saved_grids */
    hsize_t new_total = start_idx + (hsize_t)nsave;
    H5Dwrite(d_tot, H5T_NATIVE_HSIZE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &new_total);

    free(buf);
    H5Dclose(d_grids);
    H5Dclose(d_attrs);
    H5Dclose(d_tot);
    H5Pclose(dcpl);
    H5Fclose(file_id);

    return 0;
}
