/* =============================================================================
 * PPPPPP     y   y             GGGGG         AAAAA        SSSSS        PPPPP
 * P    P      y y              G             A   A        S            P   P
 * PPPPPP       y               G  GG         AAAAA        SSSSS        PPPPP
 * P            y               G   G         A   A            S        P
 * P.           y     _____     GGGGG         A   A        SSSSS        P
 * =============================================================================
 * 
 * GPU Accelerated Spin Physics
 * 
 * Python Extension
 * -----------------------------------------------------------------------------
 * Description : 
 *  This extension provides a Python interface for the G A S P (GPU Accelerated 
 *  Spin Physics) code, enabling efficient sampling of many 2D Ising model replicas
 *  on a GPU. See documentation at <documentation placeholder>.
 *   
 * Author      : David Quigley (University of Warwick)
 * Version     : 0.1
 * Date        : September 2025
 *
 * -----------------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>  
#include <float.h>
#include <stdbool.h>
#include <stdint.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <structmember.h>

extern "C" {
  #include "io.h"
  #include "grid.h"
  #include "bootstrap.h"
}

#include "mc_gpu.h"

/* ------------------------------------------------
 * Module level variables controlling GPU execution 
 * ----------------------------------------------*/

bool run_gpu = true;    // Run using GPU
bool run_cpu = false;   // Run using CPU

int idev = -1; // GPU device to use
int gpu_nsms;  // Number of multiprocessors on the GPU

/* -----------------------------------------------
 * Module level storage for sampled grids and CVs 
 * ----------------------------------------------*/

 int8_t* grid_history = NULL;  // Pointer to memory in which we store sampled grids 
                               // generated to pass back to Python. 8 bit integers to save RAM
int ihist = 0;                 // Current snapshot number
int maxhist = NULL;            // Maximum number of snapshot to allocate space for

/* Also defined on initialisation (PyInit_gasp) are lists of magnetisation for each grid at 
   each mag_output_int, and (if specified) the largest cluster size. */

/* ************************************* 
 * DEFINITION OF GRID SNAPSHOT OBJECT
 * *************************************

/* Struct which stores the underlying data of the grid snapshot object */
typedef struct {
    PyObject_HEAD
    int isweep;             // MC sweep at which the grid was captured
    int igrid;              // Which grid in the ensemble is this
    float magnetisation;    // Sum of spins
    float lclus_size;       // Size of largest cluster of spins opposite to initial spin
    float committor;        // Committor value
    PyObject *grid;         // Numpy array that holds the grid of spins
} GridSnapObject;

/* Struct which defines the members of the struct as they'll be seen by Python */
static PyMemberDef GridSnapObject_members[] = {
    {"isweep", T_INT, offsetof(GridSnapObject, isweep), 0, "MC sweep at which the grid was captured"},
    {"igrid", T_INT, offsetof(GridSnapObject, igrid), 0, "Which grid in the ensemble is this"},
    {"magnetisation", T_FLOAT, offsetof(GridSnapObject, magnetisation), 0, "Magnetisation - i.e. sum of spins"},
    {"lclus_size", T_FLOAT, offsetof(GridSnapObject, lclus_size), 0, "Size of largest cluser of spins opposite to initial spin"},
    {"committor", T_FLOAT, offsetof(GridSnapObject, committor), 0, "Committor value if available. -1 if not."},
    //{"grid", T_OBJECT, offsetof(GridSnapObject, grid), 0, "Numpy array that holds the grid of spins"},
    {NULL}  /* Sentinel */
};

/* We don't let Python directly access the grid via the struct, instead we define get/set
   functions so we can handle reference counting */

/* Getter for the grid property of grid snapshot objects */
static PyObject *get_grid(GridSnapObject *self, void *closure) {
    Py_INCREF(self->grid); // Increment reference count before returning
    return self->grid;
}

/* Setter for the grid property of grid snapshot objects */
static int set_grid(GridSnapObject *self, PyObject *value, void *closure) {
    if (!PyArray_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "grid must be a numpy array");
        return -1;
    }
    Py_XDECREF(self->grid); // Decrement old reference count
    self->grid = value;
    Py_INCREF(self->grid); // Increment new reference count
    return 0;
}

/* Register the set and get functions above */
static PyGetSetDef GridSnapObject_getset[] = {
    {"grid", (getter)get_grid, (setter)set_grid, "2D numpy array of 8-bit integers", NULL},
    {NULL}  /* Sentinel */
};

/* Constructor for the GridSnapObject type */
static int GridSnapObject_init(GridSnapObject *self, PyObject *args, PyObject *kwds) {
    long int in_isweep, in_igrid;
    PyObject *in_numpy_array = NULL;

    // A format string for parsing arguments: "iiO" means two long integers and one Python object
    if (!PyArg_ParseTuple(args, "iiO", &in_isweep, &in_igrid, &in_numpy_array)) {
        return -1; // Parsing failed
    }

    // Check if the provided object is a NumPy array of the correct type and dimensions
    if (!PyArray_Check(in_numpy_array)) {
        PyErr_SetString(PyExc_TypeError, "Third argument must be a NumPy array.");
        return -1;
    }

    // Check for 8-bit integers (np.int8) and 2 dimensions
    PyArrayObject *arr = (PyArrayObject *)in_numpy_array;
    if (PyArray_TYPE(arr) != NPY_INT8 || PyArray_NDIM(arr) != 2) {
        PyErr_SetString(PyExc_TypeError, "NumPy array must be 2D with 8-bit integer elements.");
        return -1;
    }

    // Set the properties of the C struct
    self->isweep = in_isweep;
    self->igrid = in_igrid;
    self->magnetisation = 0.0; // Default value (will always be available)
    self->lclus_size = -1.0;   // Default value (-1 if not available)
    self->committor = -1.0;    // Default value (-1 if not available)

    // Assign the NumPy array and manage its reference count
    Py_INCREF(in_numpy_array); // Increment the reference count for the new object
    self->grid = in_numpy_array;

    return 0; // Success
}

/* Deallocator for the GridSnapObject type */
static void GridSnapObject_dealloc(GridSnapObject *self) {

    // Release the reference to the NumPy grid
    Py_XDECREF(self->grid);

    // Call the base type's deallocator to free the object's memory
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// Forward declaration
static PyObject* GridSnapObject_deepcopy(PyObject* self, PyObject* memo);

// Add __deepcopy__ to GridSnapObjectType
static PyMethodDef GridSnapObject_methods[] = {
  {"__deepcopy__", (PyCFunction)GridSnapObject_deepcopy, METH_VARARGS, "Deep copy a GridSnapObject."},
  {NULL, NULL, 0, NULL}
};

/* Define the Types that GridSnapObjects are an instance of */
PyTypeObject GridSnapObjectType = {
    PyObject_HEAD_INIT(NULL)
  .tp_name = "gasp.GridSnapObject",
  .tp_basicsize = sizeof(GridSnapObject),
  .tp_dealloc = (destructor)GridSnapObject_dealloc,
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_methods = GridSnapObject_methods,
  .tp_members = GridSnapObject_members,
  .tp_getset = GridSnapObject_getset,
  .tp_init = (initproc)GridSnapObject_init,
  .tp_new = PyType_GenericNew,
};

// __deepcopy__ implementation for GridSnapObject
static PyObject* GridSnapObject_deepcopy(PyObject* self, PyObject* memo) {
  GridSnapObject* orig = (GridSnapObject*)self;
  PyObject* grid_copy = PyObject_CallMethod(orig->grid, "copy", NULL);
  if (!grid_copy) return NULL;
  PyObject* args = Py_BuildValue("iiO", orig->isweep, orig->igrid, grid_copy);
  PyObject* new_obj = PyObject_CallObject((PyObject*)&GridSnapObjectType, args);
  Py_DECREF(args);
  Py_DECREF(grid_copy);
  if (!new_obj) return NULL;
  // Copy attributes
  PyObject_SetAttrString(new_obj, "magnetisation", PyFloat_FromDouble((double)orig->magnetisation));
  PyObject_SetAttrString(new_obj, "lclus_size", PyFloat_FromDouble((double)orig->lclus_size));
  PyObject_SetAttrString(new_obj, "committor", PyFloat_FromDouble((double)orig->committor));
  return new_obj;
}

/* *************************************** 
 * END DEFINITION OF GRID SNAPSHOT OBJECT
 * ***************************************/

/* -----------------------------------------------------------------------------------------------
 * Function to reset the module level store of grid snapshots, magnetisation, and largest_cluster
 * ----------------------------------------------------------------------------------------------- */
static PyObject* reset_grids_list(PyObject* self, PyObject* args) {
    
  PyObject *module, *new_list, *new_magnetisation_list, *new_largest_cluster_list;

  // Import the module
  module = PyImport_ImportModule("gasp");
  if (!module) {
    PyErr_SetString(PyExc_ImportError, "Could not import 'gasp' module");
    return NULL;
  }

  // Create a new empty list
  new_list = PyList_New(0);
  if (!new_list) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create new list for grids");
    return NULL;
  }

  // Replace the module-level attribute
  if (PyModule_AddObject(module, "grids", new_list) < 0) {
    Py_DECREF(new_list);
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to module");
    return NULL;
  }


  // Replace the module level magnetisation list with a new list
  new_magnetisation_list = PyList_New(0);
  if (!new_magnetisation_list) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create new list for magnetisation");
    return NULL;
  }

  if (PyModule_AddObject(module, "magnetisation", new_magnetisation_list) < 0) {
    Py_DECREF(new_magnetisation_list);
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'magnetisation' attribute to module");
    return NULL;
  }

  // Replace the module level largest_cluster list with a new list
  new_largest_cluster_list = PyList_New(0);
  if (!new_largest_cluster_list) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create new list for largest_cluster");
    return NULL;
  }

  if (PyModule_AddObject(module, "largest_cluster", new_largest_cluster_list) < 0) {
    Py_DECREF(new_largest_cluster_list);
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'largest_cluster' attribute to module");
    return NULL;
  }

  Py_RETURN_NONE;
}

/* -----------------------------------------------------------------------------------------------
 * Function to update module level lists of magnetisation and largest cluster size (if computed)
 * ----------------------------------------------------------------------------------------------- */
int append_grids_list(int L, int ngrids, int* grid_data, int isweep, float* magnetisation,
                      float *lclus_size, char *cv,double dn_thr, double up_thr, char *filename) {

  // Update module level lists of magnetisation and largest cluster (if computed)
  float *cv_val = NULL;
  if (strcmp(cv, "magnetisation") == 0) {
    cv_val = magnetisation;
  } else if (strcmp(cv, "largest_cluster") == 0) {
    cv_val = lclus_size;
  } 
  if (!cv_val) {
    PyErr_SetString(PyExc_ValueError, "Invalid collective variable specified");
    return -1;
  }

  // Get a handle to the gasp module
  PyObject* module = PyImport_AddModule("gasp"); 
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Could not import module 'gasp' when appending to record lists");
    return -1;
  }
  
  // Convert C array of current grid magnetisations to a Python list
  PyObject* magnetisation_pylist = PyList_New(ngrids);
  if (!magnetisation_pylist) {
    Py_DECREF(module);
    PyErr_SetString(PyExc_RuntimeError, "Could not create Python list for magnetisation");
    return -1;
  }
  for (int i = 0; i < ngrids; ++i) {
    PyObject* val = PyFloat_FromDouble((double)magnetisation[i]);
    if (!val) {
      Py_DECREF(module);
      Py_DECREF(magnetisation_pylist);
      PyErr_SetString(PyExc_RuntimeError, "Could not create Python float object for magnetisation list");
      return -1;
    }
    PyList_SET_ITEM(magnetisation_pylist, i, val);
  }

  // Append to module-level magnetisation list
  if (module) {
    PyObject* magnetisation_list = PyObject_GetAttrString(module, "magnetisation");
    if (magnetisation_list && PyList_Check(magnetisation_list)) {
      PyList_Append(magnetisation_list, magnetisation_pylist);
    }

    Py_XDECREF(magnetisation_list);
  }
  Py_DECREF(magnetisation_pylist);

  if (lclus_size) { // We don't always record largest cluster size

    // Convert C array lclus_size to Python list
    PyObject* largest_cluster_pylist = PyList_New(ngrids);
    if (!largest_cluster_pylist) {
      return -1;
    }
  
    for (int i = 0; i < ngrids; ++i) {
      PyObject* val = PyFloat_FromDouble((double)lclus_size[i]);
      if (!val) {
        Py_DECREF(largest_cluster_pylist);
        return -1;
      }
      PyList_SET_ITEM(largest_cluster_pylist, i, val);
    }

    // Append to module-level largest_cluster list
    if (module) {
      PyObject* largest_cluster_list = PyObject_GetAttrString(module, "largest_cluster");
      if (largest_cluster_list && PyList_Check(largest_cluster_list)) {
        PyList_Append(largest_cluster_list, largest_cluster_pylist);
      }
      Py_XDECREF(largest_cluster_list);
    }
    Py_DECREF(largest_cluster_pylist);

  } // end if recording largest cluster size

  // Always call the hdf5 writer
  int iret = write_ising_grids_hdf5(L, ngrids, grid_data, isweep, magnetisation, lclus_size, cv, dn_thr, up_thr, filename);

  if (iret != 0) {
    PyErr_SetString(PyExc_RuntimeError, "Error writing HDF5 file");
    return -1;
  }

  // If we're maintaining a history of grids in RAM then proceed, otherwise we're done
  if (!grid_history || ihist >= maxhist) {
    return iret;
  }

  // Dimensions of NumPy array member of GridSnapObjects
  npy_intp dims[2] = {L, L};

  // Create a Python list to hold GridSnapObject instances corresponding to the current sweep
  PyObject* grid_snap_list = PyList_New(0);
  if (!grid_snap_list){
    PyErr_SetString(PyExc_RuntimeError, "Could not create Python list for current grid snapshots");
    return -1;
  }

  int snapsize = L*L;
  for (int i = 0; i < ngrids; ++i) {

    // Only do this if within thresholds
    // Append if the selected CV lies between up_thr and down_thr
    if ( ((double)cv_val[i] > dn_thr) && ((double)cv_val[i] < up_thr) ) {

      // Copy current grid data into history array. Not using memcpy as there's a cast involved.
      int isite;
      for (isite=0;isite<L*L;++isite){
        grid_history[snapsize*ihist + isite] = (int8_t)grid_data[i*L*L+isite];
      } 
    
      // Create a NumPy array from the data. This just wraps the part of grid_history
      // we just populated.
      PyObject* array = PyArray_SimpleNewFromData(2, dims, NPY_INT8, (int8_t*)(grid_history + ihist*snapsize));
      if (!array) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create NumPy array from grid\n");
        Py_DECREF(grid_snap_list);
        return -1;
      }

      Py_INCREF(Py_None);
      PyArray_SetBaseObject((PyArrayObject*)array, Py_None);
    
      // Create GridSnapObject instance
      PyObject* args = Py_BuildValue("iiO", isweep, i, array);
      PyObject* snap_obj = PyObject_CallObject((PyObject*)&GridSnapObjectType, args);
      Py_DECREF(args);
      Py_DECREF(array);  // We're done with the array we passed into the GridSnapObject constructor
      if (!snap_obj) {
        PyErr_SetString(PyExc_RuntimeError, "Could not create GridSnapObject instance for current grid\n");
        Py_DECREF(grid_snap_list);
        return -1;
      }

      // Set additional attributes of GridSnapObject instance
      PyObject_SetAttrString(snap_obj, "magnetisation", PyFloat_FromDouble((double)magnetisation[i]));
      if (lclus_size) {
        PyObject_SetAttrString(snap_obj, "lclus_size", PyFloat_FromDouble((double)lclus_size[i]));
      }
      PyObject_SetAttrString(snap_obj, "committor", PyFloat_FromDouble(-1.0)); // Default value

      // Append current GridSnapObject instance to the list for the current sweep
      PyList_Append(grid_snap_list, snap_obj);
      ihist++;

      if (ihist == maxhist) {
        PySys_WriteStderr("\nWARNING: Maximum history reached, not storing more snapshots to grids object!\n\n");
        ihist++;
        break; // Don't try and store any more grids from this iteration
      } 
    
    } // if between thresholds

  } //grids

  // Retrieve existing module-level list of grids
  PyObject* existing_list = PyObject_GetAttrString(module, "grids");
  if (!existing_list || !PyList_Check(existing_list)) {
    Py_XDECREF(existing_list);
    Py_DECREF(grid_snap_list);
    PyErr_SetString(PyExc_RuntimeError, "Attribute 'grids' not found or not a list");
    return -1;
  }

  // Append the new list of GridSnapObject instances only if grid_snap_list is not empty
  if (PyList_Size(grid_snap_list) > 0 ) {
    if (PyList_Append(existing_list, grid_snap_list) < 0) {
      Py_DECREF(existing_list);
      Py_DECREF(grid_snap_list);
      return -1;
    }
  }

  Py_DECREF(existing_list);
  Py_DECREF(grid_snap_list);
  
  return 0;
  
}



/* ===========================================================
   r u n _ n u c l e a t i o n _ s w a r m
   --------------------------------
   Run a swarm of nucleation simulations on either CPU or GPU
   ============================================================*/
/* Define documentation string for the function */
char *nuc_docstring = 
"run_nucleation_swarm(L, ngrid, tot_nsweeps, beta, h, initial_spin=-1, cv='magnetisation', up_threshold=-0.9*initial_spin, \n\
                      dn_threshold=0.95*initial_spin, mag_output_int=100, grid_output_int=1000, keep_grids=True, \n\
                      max_keep_grids=0, threadsPerBlock=32, gpu_method=0)\n\
\n\
Simulate a swarm of 2D Ising model realisations (grids) to capture nucleation events using GPU acceleration.\n\
Returns information on fraction of nucleated realisations versus time and populates a module-level list of grid snapshots \n\
captured between specified thresholds of a chosen collective variable (CV). A module level history of magnetisation is \n\
also populated, as is largest_cluster if selected as the CV.\n\
\n\
Parameters\n\
----------\n\
L : int\n\
    Size of each 2D Ising grid (L x L).\n\
ngrid : int\n\
    Number of independent grids (realisations) to simulate.\n\
tot_nsweeps : int\n\
    Total number of Monte Carlo sweeps per grid before exiting even if not all are nucleated.\n\
beta : float\n\
    Inverse temperature.\n\
h : float\n\
    Magnetic field.\n\
initial_spin : int, optional\n\
    Initial majority spin in the parent phase (default: -1).\n\
cv : str, optional\n\
    Collective variable (CV) for nucleation detection ('magnetisation' or 'largest_cluster').\n\
up_threshold : float, optional\n\
    CV threshold for nucleation detection (default: depends on initial_spin).\n\
dn_threshold : float, optional\n\
    CV Threshold for return to parent phase (default: depends on initial_spin).\n\
mag_output_int : int, optional\n\
    Sweeps between magnetisation/CV sampling (default: 100).\n\
grid_output_int : int, optional\n\
    Sweeps between grid output to HDF5 (default: 1000).\n\
keep_grids : bool, optional\n\
    Whether to keep grid history in RAM by populating the module level grids object (default: True).\n\
max_keep_grids : int, optional\n\
    Maximum number of grids to keep (default: ngrids*nsweeps//grid_ouptut_int).\n\
threadsPerBlock : int, optional\n\
    Number of GPU threads (replicas) to launch per gread block (default: 32).\n\
gpu_method : int, optional\n\
    GPU kernel method to use (default: 0).\n\
\n\
Returns\n\
-------\n\
numpy.ndarray\n\
    Fraction of nucleated trajectories at each mag_output_int.";

/* The actual function */
static PyObject* method_run_nucleation_swarm(PyObject* self, PyObject* args, PyObject* kwargs){

  /* Seed for random number generator from the clock */
  unsigned long rngseed = (long)time(NULL);
  
  /* Positional arguments to extract and their defaults */
  int L = 64;             // Size of LxL grid
  int ngrids = 128;       // Number of grids in the swarm
  int tot_nsweeps = 100;  // Number of sweeps to run for each grid

  double beta = 0.54;     // inverse temperature
  double h = 0.07;        // magnetic field


  /* Keyword arguments to extract and their defaults */
  int initial_spin = -1;                             // Majority spin in parent phase
   
  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.95*(double)initial_spin;  // Threshold mag at which assumed returned

  int mag_output_int = 100;                          // Sweeps between output of magnetisation
  int grid_output_int = 1000;                        // Sweeps between output of grids
  int keep_grids = 1;                                // Keep history of grids in RAM
  int max_keep_grids = 0;                            // Maximum number of grids to allocate space for

  int threadsPerBlock = 32;                          // Threads per block
  int gpu_method = 0;                                // GPU method to use - see mc_gpu.cu

  char *cv="magnetisation";                          // Collective variable to use in determining fate

  char *dynamics = "spin_flip";                      // Dynamics to use - currently only spin flip or kawasaki.

  /* list of keywords */
  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h", "dynamics",
    "initial_spin", "cv", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "keep_grids", "max_keep_grids", "threadsPerBlock", "gpu_method", NULL};

  /* Parse the input tuple */
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|sisddiipiii", kwlist,
                   &L, &ngrids, &tot_nsweeps, &beta, &h, &dynamics, &initial_spin, &cv,
                   &up_threshold, &dn_threshold, &mag_output_int,
                   &grid_output_int, &keep_grids, &max_keep_grids, &threadsPerBlock, &gpu_method)) {
    return NULL;
   }

  /* Validate dynamics argument */
  if (strcmp(dynamics, "spin_flip") != 0 && strcmp(dynamics, "kawasaki") != 0) {
    PyErr_SetString(PyExc_ValueError, "dynamics must be either 'spin_flip' or 'kawasaki'");
    return NULL;
  }

  /* Validate cv argument */
  if (strcmp(cv, "magnetisation") != 0 && strcmp(cv, "largest_cluster") != 0) {
    PyErr_SetString(PyExc_ValueError, "cv must be either 'magnetisation' or 'largest_cluster'");
    return NULL;
  }

  /* Sanity check that CV thresholds are sensible if using cluster size rather than magnetisation */
  if (strcmp(cv, "largest_cluster") == 1) {
    // Warn if magnitude of up_threshold and dn_threshold are less than 2
    if (fabs(up_threshold) < 2.0 || fabs(dn_threshold) < 2.0) {
      PyErr_SetString(PyExc_UserWarning, "For cv='largest_cluster', up_threshold and dn_threshold should have magnitudes >= 2");
    }
  }

  /* Check that max_keep_grids in a positive integer */
  if (max_keep_grids < 0) {
    PyErr_SetString(PyExc_ValueError, "max_keep_grids must be a positive integer");
    return NULL;
  }

/* Delete old output and allocate space for grid storage 
   ----------------------------------------------------- */
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  /* Free grid_history if allocated */
  if (grid_history != NULL) { free(grid_history); }

  /* Determine number of grids to allocate space for */
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  if (max_keep_grids){
    maxhist = max_keep_grids;
  } else {
    maxhist = nsnaps*ngrids;
  }

  /* Allocate space for grid_history */
  if (keep_grids) {
    grid_history = (int8_t *)malloc(maxhist*L*L*sizeof(int8_t));
    if (grid_history == NULL){
      PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
      return NULL;   
    }
  }

  /* Initialise module level history of stored grids */
  ihist = 0;
  
/* Initialise simulations
   ---------------------- */

  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  /* Initialise as 100% spin down for all grids */
  ising_grids = init_grids_uniform(L, ngrids, initial_spin);
  grid_fate = NULL ; // not used

  /* Allocate space for the array representing fraction nucleated against time */
  float *result = (float *)malloc((tot_nsweeps/mag_output_int) * sizeof(float)); 
  if (result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for result array");
    free(ising_grids);
    if (grid_fate) free(grid_fate);
    return NULL;
  }

/* Run simulations - CPU version
   ----------------------------- */

  if (run_cpu==true) {

 #ifndef PYTHON
    fprintf(stdout,"Using CPU\n");
#endif
    
    /* Initialise host RNG */
    init_genrand(rngseed);

    /* Precompute acceptance probabilities for flip moves */
    preComputeProbs_cpu(beta, h);

    /* Pack arguments into structures */
    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 0; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; calc.ninputs = 1; calc.result = result;
    calc.filename="gridstates.hdf5"; calc.dynamics = dynamics;

    /* Write stdout output header*/
#ifndef PYTHON
    PySys_WriteStdout(stdout, "# isweep    nucleated fraction\n");
#endif

    /* Create HDF5 file and write attributes */
    create_ising_grids_hdf5(L, ngrids, tot_nsweeps, h, beta, calc.itask, calc.filename);
    
    /* Perform simulations calling append_grids_list periodically */
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/* Run simulations - GPU version
   ----------------------------- */
  if (run_gpu==true){

    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    /* Initialise model grid on GPU */
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    /* Select gpu_method */
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);

    /* Initialise RNG on GPU */
    curandState *d_state;                  // Pointer to device RNG states
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    /* Precompute acceptance probabilities for flip moves */
    preComputeProbs_gpu(beta, h);

    /* Pack arguments into structures */
    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.initial_spin = initial_spin; calc.cv = cv; calc.itask = 0; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold; 
    calc.ninputs = 1; calc.result = result; calc.filename="gridstates.hdf5"; calc.dynamics = dynamics;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    /* Write stdout output header*/
#ifndef PYTHON
    PySys_WriteStdout(stdout, "# isweep    nucleated fraction\n");
#endif

    /* Create HDF5 file and write attributes */
    create_ising_grids_hdf5(L, ngrids, tot_nsweeps, h, beta, calc.itask, calc.filename);

    /* Perform simulations calling append_grids_list periodically */
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);

    /* Free device arrays */
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

 }
  
/* Clean up and return
   ------------------- */

  if (ising_grids) free(ising_grids);  // Release host arrays

  /* Return a NumPy array holding the contents of the C array 'result' */
  npy_intp dims[1] = {tot_nsweeps/mag_output_int};
  PyObject* result_array = PyArray_SimpleNew(1, dims, NPY_FLOAT);  // Create array object
  if (!result_array) {
    if (result) free(result);
    return NULL;
  }

  /* C array to hold the calculated result */
  float* result_data = (float*)PyArray_DATA((PyArrayObject*)result_array); 
  if (!result_data) {
    Py_DECREF(result_array);
    if (result) free(result);
    PyErr_SetString(PyExc_RuntimeError, "Could not allocate result array data.");
    return NULL;
  }

  /* Copy data from C array to NumPy array */
  for (int i = 0; i < dims[0]; ++i) {
    result_data[i] = result[i];
  }
  if (result) free(result);

  /* Finally return numpy array holding fraction of nucleated grids vs time */
  return result_array;

}
/* ===========================================================
   r u n _ c o m m i t t o r _ c a l c
   --------------------------------
   Run a committor calculation on either CPU or GPU
   ============================================================*/
/* Define documentation string for the function */  
char *comm_docstring = 
"run_committor_calc(L, ngrid, tot_nsweeps, beta, h, initial_spin=-1, cv='magnetisation', up_threshold=-0.9*initial_spin, \n\
                   dn_threshold=0.93*initial_spin, mag_output_int=100, grid_output_int=1000, keep_grids=True, \n\
                   max_keep_grids=0, threadsPerBlock=32, gpu_method=0, grid_input='gridstates.bin', grid_array=None, nsms=gpu_nsms)\n\
\n\
Perform a committor calculation on a set of 2D Ising model grid configurations using GPU acceleration.\n\
The grid configurations can be read from a binary file or passed in as a list of 2D NumPy arrays. The function \n\
returns an array of committor probabilities corresponding to each input grid configuration. A module level history of \n\
grid snapshots is also populated, as is magnetisation and largest_cluster if selected as the CV.\n\
\n\
Parameters\n\
----------\n\
L : int\n\
    Size of each 2D Ising grid (L x L).\n\
ngrid : int\n\
    Number of independent grids (replicas) to simulate.\n\
tot_nsweeps : int\n\
    Total number of Monte Carlo sweeps per grid before exiting even if not all grids have reached state A or B.\n\
beta : float\n\
    Inverse temperature.\n\
h : float\n\
    Magnetic field.\n\
initial_spin : int, optional\n\
    Initial majority spin in the parent phase (default: -1).\n\
cv : str, optional\n\
    Collective variable for committor calculation ('magnetisation' or 'largest_cluster').\n\
up_threshold : float, optional\n\
    Threshold for nucleation (default: -0.9*initial_spin).\n\
dn_threshold : float, optional\n\
    Threshold for return to parent phase (default: 0.9*initial_spin).\n\
mag_output_int : int, optional\n\
    Sweeps between magnetisation output (default: 100).\n\
grid_output_int : int, optional\n\
    Sweeps between grid output (default: 1000).\n\
keep_grids : bool, optional\n\
    Whether to keep grid history in RAM (default: True).\n\
max_keep_grids : int, optional\n\
    Maximum number of grids to keep (default: 0).\n\
threadsPerBlock : int, optional\n\
    Number of GPU threads per block (default: 32).\n\
gpu_method : int, optional\n\
    GPU kernel method to use (default: 0).\n\
grid_input : str, optional\n\
    Path to input file for grid states or \"NumPy\" if passing grid_array (default: 'gridstates.bin').\n\
grid_array : list of numpy.ndarray, optional\n\
    List of initial grid states to compute pB for as list of 2D NumPy arrays.\n\
nsms : int, optional\n\
    Number of GPU multiprocessors to use. Defaults to the number of available multiprocessors.\n\
\n\
Returns\n\
-------\n\
list of tuple\n\
    List of (committor, grid_index) tuples for each input grid.";

/* The actual function */
static PyObject* method_run_committor_calc(PyObject* self, PyObject* args, PyObject* kwargs){

  /* Seed for random number generator from the clock */
  unsigned long rngseed = (long)time(NULL);

  /* Positional arguments to extract and their defaults */
  int L = 64;             // Size of LxL grid
  int ngrids = 128;       // Number of grids in the swarm
  int tot_nsweeps = 100;  // Number of sweeps to run for each grid

  double beta = 0.54;     // inverse temperature
  double h = 0.07;        // magnetic field

  /* Keyword arguments to extract and their defaults */
  int initial_spin = -1;                             // Majority spin in parent phase

  double up_threshold = -0.90*(double)initial_spin;  // Threshold mag at which which assumed reversed
  double dn_threshold =  0.93*(double)initial_spin;  // Threshold mag at which assumed returned
  
  int mag_output_int = 100;                          // Sweeps between output of magnetisation
  int grid_output_int = 1000;                        // Sweeps between output of grids
  int keep_grids = 1;                                // Keep history of grids in RAM
  int max_keep_grids = 0;                            // Maximum number of grids to allocate space for

  int threadsPerBlock = 32;                          // Threads per block
  int gpu_method = 0;                                // GPU method to use - see mc_gpu.cu

  char* cv = "magnetisation";                        // Collective variable to use in determining fate
  char* dynamics = "spin_flip";                      // Dynamics to use - currently only spin flip or kawasaki.

  const char* grid_input = "gridstates.bin";         // Input file for grids or "NumPy" if passing grid_array
  PyObject* grid_array_obj = NULL;                   // List of NumPy arrays for initial grids

  int nsms = gpu_nsms;                               // Number of GPU multiprocessors to use (all by default)

  /* list of keywords */
  static char* kwlist[] = {"L", "ngrid", "tot_nsweeps", "beta", "h", "dynamics",
    "initial_spin", "cv", "up_threshold", "dn_threshold", "mag_output_int",
    "grid_output_int", "keep_grids", "max_keep_grids", "threadsPerBlock", "gpu_method", "grid_input", "grid_array", "nsms", NULL};

  /* Parse the input tuple */ 
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iiidd|sisddiipiiisOi", kwlist,
       &L, &ngrids, &tot_nsweeps, &beta, &h, &dynamics, &initial_spin, &cv,
       &up_threshold, &dn_threshold, &mag_output_int,
       &grid_output_int, &keep_grids, &max_keep_grids, &threadsPerBlock, &gpu_method, &grid_input, &grid_array_obj, &nsms)) {
    return NULL;
  }

  /* Validate dynamics argument */
  if (strcmp(dynamics, "spin_flip") != 0 && strcmp(dynamics, "kawasaki") != 0) {
    PyErr_SetString(PyExc_ValueError, "dynamics must be either 'spin_flip' or 'kawasaki'");
    return NULL;
  }

  /* Validate nsms argument */
  if (nsms < 0) {
    PyErr_SetString(PyExc_ValueError, "nsms must be a non-negative integer");
    return NULL;
  } else {
    gpu_nsms = nsms;
  }

  /* Validate cv argument */
  if (strcmp(cv, "magnetisation") != 0 && strcmp(cv, "largest_cluster") != 0) {
    PyErr_SetString(PyExc_ValueError, "cv must be either 'magnetisation' or 'largest_cluster'");
    return NULL;
  }

  /* Check that max_keep_grids is an positive integer */
  if (max_keep_grids < 0) {
    PyErr_SetString(PyExc_ValueError, "max_keep_grids must be a non-negative integer");
    return NULL;
  }

  /* Validate grid_input argument */
  int grid_array_count = 1; // Default to 1 if no grid_array provided
  if (grid_array_obj && grid_array_obj != Py_None) {
    if (!PyList_Check(grid_array_obj)) {
      PyErr_SetString(PyExc_TypeError, "grid_array must be a list of 2D NumPy arrays");
      return NULL;
    }
    grid_array_count = (int)PyList_Size(grid_array_obj);
    if (grid_array_count > gpu_nsms) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length exceeds gpu_nsms");
      return NULL;
    }
    if (grid_array_count == 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list is empty");
      return NULL;
    }
    if (gpu_nsms % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "grid_array list length must divide gpu_nsms exactly");
      return NULL;
    }
    if (ngrids % gpu_nsms != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of gpu_nsms");
      return NULL;
    }
    if (ngrids % grid_array_count != 0) {
      PyErr_SetString(PyExc_ValueError, "ngrids must be a multiple of grid_array list length");
      return NULL;
    }
  }

  /* Create C array of input grids to use from Python list of NumPy arrays */
  int *grid_array_c = NULL;   
  grid_array_c = (int *)malloc(grid_array_count * L * L * sizeof(int));
  if (!grid_array_c) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for grid_array_c");
    return NULL;
  }
  for (int arr_idx = 0; arr_idx < grid_array_count; ++arr_idx) {
    PyObject* arr_obj = PyList_GetItem(grid_array_obj, arr_idx);
    if (!PyArray_Check(arr_obj)) {
      PyErr_SetString(PyExc_TypeError, "All items in grid_array must be NumPy arrays");
      free(grid_array_c);
      return NULL;
    }
    PyArrayObject* arr = (PyArrayObject*)arr_obj;
    if (PyArray_TYPE(arr) != NPY_INT8 || PyArray_NDIM(arr) != 2) {
      PyErr_SetString(PyExc_TypeError, "Each grid_array item must be a 2D NumPy array of type int8");
      free(grid_array_c);
      return NULL;
    }
    int nrows = (int)PyArray_DIM(arr, 0);
    int ncols = (int)PyArray_DIM(arr, 1);
    if (nrows != L || ncols != L) {
      PyErr_SetString(PyExc_ValueError, "Each grid_array item must have shape LxL");
      free(grid_array_c);
      return NULL;
    }
    npy_int8* arr_data = (npy_int8*)PyArray_DATA(arr);
    for (int i = 0; i < L * L; ++i) {
      grid_array_c[arr_idx * L * L + i] = (int)arr_data[i];
    }
  }
  
/* Delete old output and allocate space for output grid storage 
   ------------------------------------------------------------ */
  remove("gridstates.bin");

  /* Reset the module level list of grids to empty */
  reset_grids_list(self, NULL);

  /* Free grid_history if allocated */
  if (grid_history != NULL) { free(grid_history); }

  /* Determine number of grids to allocate space for */
  int nsnaps = tot_nsweeps/grid_output_int + 1;
  if (max_keep_grids){
    maxhist = max_keep_grids;
  } else {
    maxhist = nsnaps*ngrids;
  }

  /* Allocate space for grid_history */
  if (keep_grids) {
    grid_history = (int8_t *)malloc(maxhist*L*L*sizeof(int8_t));
    if (grid_history == NULL){
      PyErr_SetString(PyExc_MemoryError, "Error allocating RAM to hold grid history!");
      return NULL;   
    }
  }

  /* Initialise module level history of stored grids */
  ihist = 0;
  
/* Initialise simulations
   ---------------------- */
  int *ising_grids; // array of LxLxngrids spins
  int *grid_fate;   // stores pending(-1), reached B first (1) or reached A first (0)
  
  /* Initialise grids either from file or from supplied array */
  if (strcmp(grid_input, "gridstates.bin") == 0) {
    ising_grids = init_grids_from_file(L, ngrids); // read from gridinput.bin
  } else if ((strcmp(grid_input, "NumPy") == 0) && grid_array_obj != NULL) {
    ising_grids = init_grids_from_array(L, ngrids, grid_array_count, grid_array_c); // read from the supplied array
  } else {
    PyErr_SetString(PyExc_ValueError, "Invalid grid_input option or no grid_array supplied");
    return NULL;
  }

  /* Initialise grid fates - flag for whether each grid reached A or B first */
  grid_fate = init_fates(ngrids); // grid fates

  /* Allocate space for the array representing committor results */
  float *result = (float *)malloc(grid_array_count * sizeof(float)); 
  if (result == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for result array");
    free(ising_grids);
    if (grid_fate) free(grid_fate);
    if (grid_array_c) free(grid_array_c);
    return NULL;
  }

  /* Allocate space for the array representing error bars on committor estimates */
  double *errbar = (double *)malloc(grid_array_count * sizeof(double)); 
  if (errbar == NULL) {
    PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for error bar array");
    free(ising_grids);
    free(grid_fate);
    if (grid_array_c) free(grid_array_c);
    free(result);
    return NULL;
  }

/* Run simulations - CPU version
   ----------------------------- */

  if (run_cpu==true) {

#ifndef PYTHON
    fprintf(stdout,"Using CPU\n");
#endif

    /* Initialise host RNG */
    init_genrand(rngseed);

    /* Precompute acceptance probabilities for flip moves */
    preComputeProbs_cpu(beta, h);

    /* Pack arguments into structures */  
    mc_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.initial_spin = initial_spin; calc.cv = cv; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold;
    calc.ninputs = grid_array_count; calc.result=result; calc.filename = "pBgrids.hdf5"; calc.dynamics = dynamics;

    /* Create HDF5 file and write attributes */
    create_ising_grids_hdf5(L, ngrids, tot_nsweeps, h, beta, calc.itask, calc.filename);

    /* Perform the MC simulations calling append_grids_list periodically */
    mc_driver_cpu(grids, beta, h, grid_fate, samples, calc, append_grids_list);
    
  }

/* Run simulations - GPU version
   ----------------------------- */
  if (run_gpu==true){

    int *d_ising_grids;                    // Pointer to device grid configurations
    int *d_neighbour_list;                 // Pointer to device neighbour lists

    /* Initialise model grid on GPU */
    gpuInitGrid(L, ngrids, threadsPerBlock, ising_grids, &d_ising_grids, &d_neighbour_list); 

    /* Select gpu_method */
    gpu_method = select_gpu_method(L, ngrids, threadsPerBlock, idev);

    /* Initialise RNG on GPU */
    curandState *d_state;                  // Pointer to device RNG states
    gpuInitRand(ngrids, threadsPerBlock, rngseed, &d_state);

    /* Precompute acceptance probabilities for flip moves */
    preComputeProbs_gpu(beta, h);

    /* Pack arguments into structures */
    mc_gpu_grids_t grids; grids.L = L; grids.ngrids = ngrids; grids.ising_grids = ising_grids;
    grids.d_ising_grids = d_ising_grids; grids.d_neighbour_list = d_neighbour_list;
    mc_sampler_t samples; samples.tot_nsweeps = tot_nsweeps; samples.mag_output_int = mag_output_int; samples.grid_output_int = grid_output_int;
    mc_function_t calc; calc.itask = 1; calc.cv = cv; calc.initial_spin = initial_spin; calc.dn_thr = dn_threshold; calc.up_thr = up_threshold;
    calc.ninputs = grid_array_count; calc.result=result; calc.filename = "pBgrids.hdf5"; calc.dynamics = dynamics;
    gpu_run_t gpu_state; gpu_state.d_state = d_state;  gpu_state.threadsPerBlock = threadsPerBlock; gpu_state.gpu_method = gpu_method;

    /* Create HDF5 file and write attributes */
    create_ising_grids_hdf5(L, ngrids, tot_nsweeps, h, beta, calc.itask, calc.filename);

    /* Perform simulations calling append_grids_list periodically */
    mc_driver_gpu(grids, beta, h, grid_fate, samples, calc, gpu_state, append_grids_list);
    
    /* Free device arrays */
    gpuErrchk( cudaFree(d_state) );
    gpuErrchk( cudaFree(d_ising_grids) );
    gpuErrchk( cudaFree(d_neighbour_list) );

 }


/* Calculate bootstrap error bars on the committor estimates
   --------------------------------------------------------- */
  for (int i = 0; i < grid_array_count; ++i) {
    errbar[i] = bootstrap_errbar(ngrids/grid_array_count, ngrids/grid_array_count, grid_fate+(i * ngrids/grid_array_count));
    if (errbar[i] < 0.0) {
      PyErr_SetString(PyExc_RuntimeError, "Error calculating bootstrap error bar");
      free(ising_grids);
      free(grid_fate);
      if (grid_array_c) free(grid_array_c);
      free(result);
      free(errbar);
      return NULL;
    }    
  }

  
/* Clean up and return
   ------------------- */
  free(ising_grids);
  free(grid_fate);
  if (grid_array_c) free(grid_array_c);

  /* Convert result array to a list of Python tuples. First entry in the tuple will hold the pB estimate.
     Second entry in the tuple will hold the uncertainty on the pB estimate */
  
  /* Create list to hold the tuples */
  PyObject* pylist = PyList_New(grid_array_count);
  if (!pylist) {
    if (result) free(result);
    return NULL;
  }

  /* Loop over input grids */
  for (int i = 0; i < grid_array_count; ++i) {

    /* Create a tuple */
    PyObject* tuple = PyTuple_New(2);
    if (!tuple) {
      Py_DECREF(pylist);
      if (result) free(result);
      return NULL;
    }

    /* Populate the tuple with pB estimate and error bar */
    PyTuple_SET_ITEM(tuple, 0, PyFloat_FromDouble((double)result[i]));
    PyTuple_SET_ITEM(tuple, 1, PyFloat_FromDouble(errbar[i]));
    PyList_SET_ITEM(pylist, i, tuple);
  }

  /* Free result and errbar arrays */
  if (result) free(result);
  if (errbar) free(errbar);
  
  /* Finally return the list of tuples */
  return pylist;
}

/* -----------------------------------------------------------------
 * Definition of the gasp module, its methods and the init function
 * ----------------------------------------------------------------*/

/* List of methods exposed to Python in this module */
static PyMethodDef GPUIsingMethods[] = {
  {"run_committor_calc", (PyCFunction)method_run_committor_calc, METH_VARARGS | METH_KEYWORDS, comm_docstring},
  {"run_nucleation_swarm", (PyCFunction)method_run_nucleation_swarm, METH_VARARGS | METH_KEYWORDS, nuc_docstring},
  {NULL, NULL, 0, NULL} /* Sentinal */
};

/* Defines the Python module that our wrapped function will live inside */
static struct PyModuleDef gaspmodule = {
    PyModuleDef_HEAD_INIT,   /* Always this https://docs.python.org/3/c-api/module.html#initializing-c-modules */
    "gasp",                  /* Name of the module (same as name of wrapped function in this case */
    "Python interface to the GASP code for GPU accelerated 2D Ising Model", /* DocString for the module */
    -1,                      /* Memory required (bytes) for each instance, or -1 if multiple instances not supported */
    GPUIsingMethods,         /* Name of the PyMethodDef object to use for the list of member functions */
};

/* The init function for the module */
PyMODINIT_FUNC PyInit_gasp(void) { 

  /* Initialize NumPy API */
  import_array();
  
  /* Register the custom type we'll use to pass grids back to Python */
  if (PyType_Ready(&GridSnapObjectType) < 0) {
      return NULL;
  }

  /* Assign created module to a variable */
  PyObject* module = PyModule_Create(&gaspmodule);
  if (!module) {
    PyErr_SetString(PyExc_RuntimeError, "Could not create gasp module");
    return NULL;
  }

  /* Add the custom type to the module */
  Py_INCREF(&GridSnapObjectType);
  if (PyModule_AddObject(module, "GridSnapObject", (PyObject *) &GridSnapObjectType) < 0) {
      Py_DECREF(&GridSnapObjectType);
      Py_DECREF(module);
      return NULL;
  }

  /* Initialise GPU device if available on import */
  if (run_gpu==true) {

    /* Initialise GPU device(s) */
    idev = gpuInitDevice(-1, &gpu_nsms); 
    if (idev==-1){
      printf("Falling back to CPU\n");
      run_cpu=true;
      run_gpu=false;
    }
    
  }

  /* Create a module level object to hold the history of grids */
  PyObject* list = PyList_New(0);
  if (!list) {
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }

  /* Set the list as a module-level attribute called grids */
  if (PyModule_AddObject(module, "grids", list) < 0) {
    Py_DECREF(list);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'grids' attribute to gasp module");
    return NULL;
  }

  /* Add gpu_nsms as a module-level integer attribute */
  PyObject* py_gpu_nsms = PyLong_FromLong((long)gpu_nsms);
  if (PyModule_AddObject(module, "gpu_nsms", py_gpu_nsms) < 0) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'gpu_nsms' attribute to gasp module");
    return NULL;
  }

  /* Add module-level lists for magnetisation and largest_cluster */
  PyObject* magnetisation_list = PyList_New(0);
  if (!magnetisation_list) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }

  /* Set the list as a module-level attribute called magnetisation */
  if (PyModule_AddObject(module, "magnetisation", magnetisation_list) < 0) {
    Py_DECREF(magnetisation_list);
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'magnetisation' attribute to gasp module");
    return NULL;
  }

  /* Add module-level list for largest_cluster */
  PyObject* largest_cluster_list = PyList_New(0);
  if (!largest_cluster_list) {
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    return NULL;
  }

  /* Set the list as a module-level attribute called largest_cluster */
  if (PyModule_AddObject(module, "largest_cluster", largest_cluster_list) < 0) {
    Py_DECREF(largest_cluster_list);
    Py_DECREF(magnetisation_list);
    Py_DECREF(list);
    Py_DECREF(py_gpu_nsms);
    Py_DECREF(module);
    Py_DECREF(&GridSnapObjectType);
    PyErr_SetString(PyExc_RuntimeError, "Could not add 'largest_cluster' attribute to gasp module");
    return NULL;
  }

  /* Return the created module */
  return module;

}