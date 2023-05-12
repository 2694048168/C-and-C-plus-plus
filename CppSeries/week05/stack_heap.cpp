/**
 * @file stack_heap.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief stack and heap memory layout in C++
 * @attention dynamic memory || maloc and free || new and delete
 *
 */

#include <iostream>

struct Student
{
    char name[4];
    int born;
    bool male;
};

void foo()
{
    int *p = (int *)malloc(sizeof(int));
    return;
} // memory leak

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. dynamic memory with malloc and free
    ----------------------------------------------- */
    // stack memory 地址递减
    int a = 0;
    int b = 0;
    int c = 0;

    std::cout << &a << std::endl;
    std::cout << &b << std::endl;
    std::cout << &c << std::endl;

    // heap memory 地址递增 4-bytes
    int *ptr1 = (int *)malloc(4);
    int *ptr2 = (int *)malloc(4);
    int *ptr3 = (int *)malloc(4);

    std::cout << ptr1 << std::endl;
    std::cout << ptr2 << std::endl;
    std::cout << ptr3 << std::endl;

    free(ptr1);
    free(ptr2);
    free(ptr3);
    std::cout << "-------------------------------" << std::endl;

    /* Step 2. dynamic memory with new and delete
    ----------------------------------------------- */
    // allocate an int, default initializer (do nothing)
    int *p1 = new int;
    // allocate an int, initialized to 0
    int *p2 = new int();
    // allocate an int, initialized to 5
    int *p3 = new int(5);
    // allocate an int, initialized to 0
    int *p4 = new int{}; // C++11
    // allocate an int, initialized to 5
    int *p5 = new int{5}; // C++11

    // allocate a Student object, default initializer
    Student *ps1 = new Student;
    // allocate a Student object, initialize the members
    Student *ps2 = new Student{"Yu", 2020, 1}; // C++11

    // allocate 16 int, default initializer (do nothing)
    int *pa1 = new int[16];
    // allocate 16 int, zero initialized
    int *pa2 = new int[16]();
    // allocate 16 int, zero initialized
    int *pa3 = new int[16]{}; // C++11
    // allocate 16 int, the first 3 element are initialized to 1,2,3, the rest 0
    int *pa4 = new int[16]{1, 2, 3}; // C++11

    // allocate memory for 16 Student objects, default initializer
    Student *psa1 = new Student[16];
    // allocate memory for 16 Student objects, the first two are explicitly initialized
    Student *psa2 = new Student[16]{{"Li", 2000, 1}, {"Yu", 2001, 1}}; // C++11
    std::cout << psa2[1].name << std::endl;
    std::cout << psa2[1].born << std::endl;

    // deallocate memory
    delete p1;
    delete p2;
    delete p3;
    delete p4;
    delete p5;
    // deallocate memory
    delete ps1;
    delete ps2;

    // deallocate the memory of the array, 基本数据类型是可以的，数组
    // delete pa1;
    delete [] pa1;
    // deallocate the memory of the array
    delete[] pa2;
    delete[] pa3;
    delete[] pa4;

    // deallocate the memory of the array, and call the destructor of the first element
    delete [] psa1;
    // deallocate the memory of the array, and call the destructors of all the elements
    delete[] psa2;

    std::cout << "-------------------------------" << std::endl;

    /* Step 3. dynamic memory leak
    --------------------------------- */
    int *ptr = nullptr;

    ptr = (int *)malloc(4 * sizeof(int));
    // some statements
    ptr = (int *)malloc(8 * sizeof(int));
    // some statements
    free(ptr);
    // the first memory will not be freed,
    // and no pointer into this address, memory leak.

    for (int i = 0; i < 1024; i++)
    {
        ptr = (int *)malloc(1024 * 1024 * 1024);
    }
    printf("End\n");
    std::cout << "-------------------------------" << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ stack_heap.cpp
 * $ clang++ stack_heap.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */