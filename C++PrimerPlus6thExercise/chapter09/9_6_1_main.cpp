/**
 * @file 9_6_1_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "9_6_1_golf.hpp"

/**
 * @brief 编写C++程序, 完成多文件编译和链接过程
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Golf golf;

    set_golf(golf, "wei li", 42);
    show_golf(golf);

    handicap(golf, 66);
    show_golf(golf);

    set_golf(golf);
    show_golf(golf);

    return 0;
}