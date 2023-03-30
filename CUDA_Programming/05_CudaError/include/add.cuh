#ifndef _ADD_CUH_
#define _ADD_CUH_

const double EPSILON = 1.0e-15;
const double value_a = 1.23;
const double value_b = 2.34;
const double value_c = 3.57; /* c = a + b */

// kernel function
__global__ void add(const double *array_x, const double *array_y, 
                    double *array_z, const size_t arraySize);

// host function
__host__ void check(const double *host_arrayZ, const int arraySize);

// device function
__device__ void element_wise(const double x, const double y, double &z);

#endif /* _ADD_CUH_ */
