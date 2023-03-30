#ifndef _ADD_CUH_
#define _ADD_CUH_

#ifdef USE_DoublePrecision
    typedef double Precision;
    const Precision EPSILON(1.0e-15);
#else
    typedef float Precision;
    const Precision EPSILON(1.0e-6);
#endif

const int NUM_REPEATS = 10;
const Precision a = 1.23;
const Precision b = 2.34;
const Precision c = 3.57;

void add_cpu(const Precision *x, const Precision *y, 
            Precision *z, const size_t arraySize);

void check(const Precision *z, const size_t arraySize);

void cpu_performance(const Precision *array_X, const Precision *array_Y,
                    Precision *array_Z, const size_t arraySize);

// ------------------------------------------------------------------
__global__ void add_gpu(const Precision *d_x, const Precision *d_y, 
                        Precision *d_z, const size_t arraySize);

__device__ void element_wise(const Precision x, const Precision y, Precision &z);

__host__ void gpu_performance(const Precision *d_X, const Precision *d_Y,
                                Precision *d_Z, const size_t arraySize);

#endif /* _ADD_CUH_ */
