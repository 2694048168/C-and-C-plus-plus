/**
 * @file new_delete.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief new and delete Operators in C++ for Dynamic Memory.
 * C++ program to illustrate dynamic allocation and deallocation of memory
 * using new and delete
 * 
 * https://www.geeksforgeeks.org/new-and-delete-operators-in-cpp-for-dynamic-memory/
 * https://www.geeksforgeeks.org/memory-layout-of-c-program/
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // Pointer initialization to null.
    // int *p = NULL;
    int *ptr = nullptr;

    // Request memory for the variable using new operator.
    ptr = new (std::nothrow) int;
    if (!ptr) /* p = 0 | NULL | nullptr */
    {
        std::cout << "allocation of memory failed\n";
    }
    else
    {
        // Store value at allocated address
        *ptr = 42;
        std::cout << "Value of p: " << *ptr << std::endl;
    }

    // Request block of memory using new operator
    float *ptr_init = new float(75.25);
    std::cout << "Value of r: " << *ptr_init << std::endl;

    // Request block of memory of size n
    const int num_size = 5;

    int *array_mem = new (std::nothrow) int[num_size];
    if (!array_mem)
    {
        std::cout << "allocation of memory failed\n";
    }
    else
    {
        for (int i = 0; i < num_size; ++i)
        {
            array_mem[i] = i + 1;
        }
        std::cout << "Value store in block of memory: ";

        for (int i = 0; i < num_size; i++)
        {
            std::cout << array_mem[i] << " ";
        }
    }

    // freed the allocated memory
    delete ptr;
    delete ptr_init;

    // freed the block of allocated memory
    delete[] array_mem;

    return 0;
}