/**
 * @file initializer_list.cpp
 * @author Wei Li (2694048168@qq.com)
 * @brief 初始化列表 initializer list; 统一进行初始化所有类型的数据; std::initializer_list<>
 * @version 0.1
 * @date 2022-01-10
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <iostream>
#include <vector>
#include <initializer_list>

class Function_without_init_list
{
public:
    int value_a;
    int value_b;
    // 构造函数进行初始化
    Function_without_init_list(int a, int b) : value_a(a), value_b(b) {}
};

// provides a unified bridge between normal arrays and POD initialization methods
class Function_with_init_list
{
public:
    std::vector<int> vec;
    Function_with_init_list(std::initializer_list<int> list)
    {
        for (std::initializer_list<int>::iterator it = list.begin(); it != list.end(); ++it)
        {
            vec.push_back(*it);
        }
    }
};

int main(int argc, char** argv)
{
    // before C++ 11
    int arr[3] = {1, 2, 3};
    Function_without_init_list function(1, 2);
    std::vector<int> vec = {1, 2, 3, 4, 5};

    std::cout << "arr[0]: " << arr[0] << std::endl;
    std::cout << "function without init list: " << function.value_a << ", " << function.value_b << std::endl;
    for (auto it = vec.begin(); it != vec.end(); ++it)
    {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    // after C++11
    // binds the concept of the initialization list to the type and calls it std::initializer_list
    // provides a unified bridge between normal arrays and POD initialization methods
    // C++11 also provides a unform syntax for initializing arbitrary objects "{}"
    Function_with_init_list function_magic = {1, 2, 3, 4, 5};
    std::cout << "Initialization with initializer_list :\n";
    for (auto it : function_magic.vec)
    {
        std::cout << it << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
