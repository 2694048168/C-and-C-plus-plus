#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <cstdlib>
#include <iostream>

// ------------------------------------
int main(int argc, char const *argv[])
{
    int num_size = 10;
    thrust::device_vector<int> x(num_size, 0);
    thrust::device_vector<int> y(num_size, 0);
    for (size_t i = 0; i < x.size(); ++i)
    {
        x[i] = i + 1;
    }

    thrust::inclusive_scan(x.begin(), x.end(), y.begin());
    for (size_t i = 0; i < y.size(); ++i)
    {
        std::cout << y[i] << " ";
    }
    std::cout << std::endl;

    /* the device GPU for thrust library
    ------------------------------------- */
    int *d_x, *d_y;
    size_t bytesSize = sizeof(int) * num_size;
    cudaMalloc(&d_x, bytesSize);
    cudaMalloc(&d_y, bytesSize);
    
    int *h_x = (int*)malloc(bytesSize);
    for (size_t i = 0; i < num_size; ++i)
    {
        h_x[i] = i + 1;
    }
    cudaMemcpy(d_x, h_x, bytesSize, cudaMemcpyHostToDevice);

    thrust::inclusive_scan(thrust::device, d_x, d_x + num_size, d_y);

    int *h_y = (int*)malloc(sizeof(int) * num_size);
    cudaMemcpy(h_y, d_y, bytesSize, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < num_size; ++i)
    {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    // TODO: free
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);

    return 0;
}
