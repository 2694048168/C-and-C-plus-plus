/**
 * @file malloc.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-22
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

unsigned maximum = 0;

/**
 * @brief 测试 malloc 最大内存申请数量
 * 
 * 程序员的自我修养: 链接、装载和库
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    unsigned blocksize[] = {1024 * 1024, 1024, 1};
    int      i, count;
    for (i = 0; i < 3; i++)
    {
        for (count = 1;; count++)
        {
            // void *block = malloc(maximum + blocksize[i] * count);
            char *block = new char[maximum + blocksize[i] * count]();
            if (block)
            {
                maximum = maximum + blocksize[i] * count;
                // free(block);
                delete[] block;
                std::printf("Now First malloc size = %u bytes\n", maximum);
            }
            else
            {
                break;
            }
        }
        std::printf("Now Second malloc size = %u bytes\n", maximum);
    }

    std::printf("Maximum malloc size = %u bytes\n", maximum);

    return 0;
}