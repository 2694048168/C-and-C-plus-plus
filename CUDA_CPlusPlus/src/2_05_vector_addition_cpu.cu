#include <iostream>
#include <stdio.h>
#include <vector>

// Defining number of elements in array.
#define N 100

// Defining vector addition function for single core CPU.
void cpu_add(int *host_a, int *host_b, int *host_c)
{
  int tid = 0;
  while (tid < N)
  {
    host_c[tid] = host_a[tid] + host_b[tid];
    tid ++;
  }
}

int main(int argc, char **argv)
{
  int host_a[N], host_b[N], host_c[N];
  // Initializing two arrays for addition.
  for (unsigned int i = 0; i < N; ++i)
  {
    host_a[i] = 2 * i;
    host_b[i] = i;
  }

  // Calling CPU function for vector addition.
  cpu_add(host_a, host_b, host_c);
  

  // Print answer.
  // printf("Vector addition on CPU\n");
  std::cout << "Vector addition on CPU\n";
  for (unsigned int i = 0; i < N; ++i)
  {
    // printf("The sum of %d element is %d + %d = %d\n", i, host_a[i], host_b[i], host_c[i]);
    std::cout << "The sum of " << i << " element is " << host_a[i] << " + " << host_b[i] << 
                                       " = " << host_c[i] << std::endl;
  }

  
  return 0;
}