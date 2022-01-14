/**
 * @file initialization_temporary_variable.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 在 if-switch 等条件判断语句中能够直接声明和使用一个临时的局部变量
 * @version 0.1
 * @date 2022-01-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
void DisplayElement(const T &container)
{
    // for (auto element : container)
    // {
    //     std::cout << element << " ";
    // }
    for (auto element = container.begin(); element != container.end(); ++element)
    {
        std::cout << *element << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv)
{
    std::vector<int> vec = {1, 2, 3, 4};
    DisplayElement(vec);

    // since c++17, can be simplified by using "auto"
    // const std::vector<int>::iterator itr = std::find(vec.begin(), vec.end(), 2);
    auto itr = std::find(vec.begin(), vec.end(), 2);
    if (itr != vec.end())
    {
        *itr = 3;
    }
    DisplayElement(vec);

    // 直接在 if 语句内部进行临时局部变量的声明和初始化
    if (auto itr = std::find(vec.begin(), vec.end(), 3); itr != vec.end())
    {
        *itr = 4;
    }
    DisplayElement(vec);
    
    return 0;
}
