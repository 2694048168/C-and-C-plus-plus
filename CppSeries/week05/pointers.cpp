/**
 * @file pointers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief A pointer in C/C++ can be used to access a specific memory location
 *  and has very good efficiency.
 * @attention a pointer is a variable for an address;
 *  the value is an address in the memory.
 *
 */

#include <iostream>
#include <cstring>

typedef struct _Student
{
    char name[4];
    int born;
    bool male;
} Student;

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. the pointer in C++
    -------------------------------- */
    int num_seed = 42;
    int *ptr1 = nullptr;
    int *ptr2 = nullptr;
    ptr1 = &num_seed;
    ptr2 = &num_seed;
    *ptr1 = 24;
    *ptr2 = 66;
    std::cout << "the value of num_seed: " << num_seed << std::endl;
    std::cout << "sizeof(ptr1) = " << sizeof(ptr1) << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /* Step 2. the pointer to Struct
    ---------------------------------- */
    Student stu = {"Yu", 2000, true};
    Student *pStu = &stu;

    std::cout << stu.name << " was born in " << stu.born
              << ". Gender: " << (stu.male ? "male" : "female") << std::endl;

    strncpy(pStu->name, "Li", 4);
    pStu->born = 2001;
    (*pStu).born = 2002;
    pStu->male = false;

    std::cout << stu.name << " was born in " << stu.born
              << ". Gender: " << (stu.male ? "male" : "female") << std::endl;

    printf("Address of stu: %p\n", pStu);                 // C style
    std::cout << "Address of stu: " << pStu << std::endl; // C++ style
    std::cout << "Address of stu: " << &stu << std::endl;
    std::cout << "Address of member name: " << &(pStu->name) << std::endl;
    std::cout << "Address of member born: " << &(pStu->born) << std::endl;
    std::cout << "Address of member male: " << &(pStu->male) << std::endl;

    std::cout << "sizeof(pStu) = " << sizeof(pStu) << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /* Step 3. the pointer to pinter
    ---------------------------------- */
    int num = 10;
    int *p = &num;
    int **pp = &p;
    *(*pp) = 42;
    std::cout << "the value of num = " << num << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /* Step 4. the pointer bound
    ---------------------------------- */
    int num_bound = 0;
    int * p_bound = &num_bound;

    p_bound[-1] = 2;    //out of bound
    p_bound[0] = 3;    //okay
    *(p_bound+1) = 4; //out of bound

    std::cout << "num_bound = " << num_bound << std::endl;
    std::cout << "-------------------------------" << std::endl;

    /* Step 5. the pointer with const
    ---------------------------------- */
    int num_random = 42;
    int another = 24;
    
    // You cannot change the value that p1 points to through p1
    const int * p1 = &num_random;
    // *p1 = 3; /* error */
    num_random = 3; /* okay */

    // You cannot change value of p2 (address)
    int * const p2 = &num_random;
    *p2 = 3; /* okay */
    // p2 = &another; /* error */

    //You can change neither
    const int* const p3 = &num_random;
    // *p3 = 3; /* error */
    // p3 = &another; /* error */

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ pointers.cpp
 * $ clang++ pointers.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */