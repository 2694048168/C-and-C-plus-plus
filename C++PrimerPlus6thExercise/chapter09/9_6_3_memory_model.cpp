/**
 * @file 9_6_3_memory_model.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

// using placement new
#include <iostream>
#include <new> // for placement new

struct Chaff
{
    char dross[20];
    int  slag;
};

/**
 * @brief 编写C++程序, 理解C++的内存模型, stack and heap
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    const unsigned BUFF_SIZE = 512;
    const unsigned NUM_SIZE  = 2;

    char buffer[BUFF_SIZE]; // chunk of memory for static memory,

    /* 1. static array on stack memory,
    ----------------------------------- */
    // through 'placement new' 指定特定的内存地址(这里是静态缓冲)存储
    Chaff *placement_ptr = new (buffer) Chaff[NUM_SIZE];

    for (size_t i = 0; i < NUM_SIZE; ++i)
    {
        const char *dross = "wei li";
        strcpy_s(placement_ptr->dross, strlen(dross) + 1, dross);
        placement_ptr->slag = 42;

        std::cout << "The dross: " << placement_ptr->dross;
        std::cout << " and the slag is " << placement_ptr->slag << "\n";

        ++placement_ptr;
    }

    /* 2. new array on heap memory,
    ----------------------------------- */
    Chaff *ptr = new Chaff[NUM_SIZE];
    for (size_t i = 0; i < NUM_SIZE; ++i)
    {
        const char *dross = "li wei";
        strcpy_s(ptr[i].dross, strlen(dross) + 1, dross);
        ptr[i].slag = 24;

        std::cout << "The dross: " << ptr[i].dross;
        std::cout << " and the slag is " << ptr[i].slag << "\n";
    }
    delete[] ptr;

    return 0;
}