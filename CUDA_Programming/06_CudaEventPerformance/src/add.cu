#include "cudaError.cuh"
#include "add.cuh"

#include <iostream>

void add_cpu(const Precision *x, const Precision *y,
             Precision *z, const size_t arraySize)
{
    for (size_t idx = 0; idx < arraySize; ++idx)
    {
        z[idx] = x[idx] + y[idx];
    }
}

void check(const Precision *z, const size_t arraySize)
{
    bool has_error = false;
    for (int n = 0; n < arraySize; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    std::cout << (has_error ? "Has errors" : "No errors") << std::endl;
}

void cpu_performance(const Precision *array_X, const Precision *array_Y,
                    Precision *array_Z, const size_t arraySize)
{
    float t_sum = 0.0f;
    float t2_sum = 0.0f;
    for (size_t repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        add_cpu(array_X, array_Y, array_Z, arraySize);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms.\n";

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    std::cout << "Time = " << t_ave << "+-" << t_err << "\n";
}

// -------------------------------------------------------------
__global__ void add_gpu(const Precision *x, const Precision *y,
                        Precision *z, const size_t arraySize)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < arraySize)
    {
        // z[idx] = x[idx] + y[idx];
        element_wise(x[idx], y[idx], z[idx]);
    }
}

__device__ void element_wise(const Precision x, const Precision y, Precision &z)
{
    z = x + y;
}

void gpu_performance(const Precision *d_X, const Precision *d_Y,
                              Precision *d_Z, const size_t arraySize)
{
    const int block_size = 128;
    const int grid_size = (arraySize + block_size - 1) / block_size;

    float t_sum = 0;
    float t2_sum = 0;
    for (int repeat = 0; repeat <= NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        add_gpu<<<grid_size, block_size>>>(d_X, d_Y, d_Z, arraySize);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        std::cout << "Time = " << elapsed_time << " ms.\n";

        if (repeat > 0)
        {
            t_sum += elapsed_time;
            t2_sum += elapsed_time * elapsed_time;
        }

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    const float t_ave = t_sum / NUM_REPEATS;
    const float t_err = sqrt(t2_sum / NUM_REPEATS - t_ave * t_ave);
    std::cout << "Time = " << t_ave << "+-" << t_err << "\n";
}
