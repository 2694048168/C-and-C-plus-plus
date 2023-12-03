/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 测试对全局变量和全局对象的使用, 保证可以被编译和链接symbol
 * @version 0.1
 * @date 2023-12-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "testGlobalVariable.h"

#include <iostream>
#include <vector>

// 使用全局变量和全局对象
#include "GlobalVariable.h"

extern int num;

extern std::map<const char *, std::vector<ImgData>> mapTable;

// ==============================
int main(int argc, char **argv)
{
    std::cout << "the init global variable: " << num << '\n';
    addOnce(42);
    std::cout << "the add operator global variable: " << num << '\n';

    std::cout << "the init global object: \n";
    for (const auto &elem : mapTable)
    {
        std::cout << elem.first << ": ";
        for (const auto &val : elem.second)
        {
            std::cout << (void *)val.pImgBuf << " " << val.nImgHeight << " " << val.nImgHeight << '\n';
        }
        std::cout << '\n';
    }

    ImgData data1{};
    data1.nImgHeight = 640;
    data1.nImgWidth  = 840;

    ImgData data2{};
    data2.nImgHeight = 840;
    data2.nImgWidth  = 640;

    std::vector<ImgData> vec{data1, data2};
    addMapTable("camera1", vec);
    addMapTable("camera2", vec);
    for (const auto &elem : mapTable)
    {
        std::cout << elem.first << ": ";
        for (const auto &val : elem.second)
        {
            std::cout << (void *)val.pImgBuf << " " << val.nImgHeight << " " << val.nImgHeight << '\n';
        }
        std::cout << '\n';
    }

    return 0;
}