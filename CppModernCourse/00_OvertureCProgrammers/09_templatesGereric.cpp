/**
 * @file 09_templatesGereric.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

/**
 * @brief Generic Programming with Templates
 * 使用模板的泛型编程, 是指写一次代码就能适用于不同的类型, 
 * 而不是通过复制和粘贴每个类型来多次重复相同的代码;
 * 在C++中, 使用模板来产生泛型代码, 模板是一种特殊的参数, 告诉编译器它代表各种可能的类型;
 * stdlib的所有容器都是模板, 在大多数情况下, 这些容器中的对象的类型并不重要,
 * 例如, 确定容器中的元素数量或返回其第一个元素的逻辑并不取决于元素的类型;
 *
 * 写一个函数, 将三个相同类型的数字相加, 希望函数接受任何可加类型;
 * 在++中, 这是一个简单的泛型编程问题, 可以直接用模板解决, 重要的代码复用.
 * 
 */

template<typename DataType>
DataType add_num(const DataType &x, const DataType &y, const DataType &z)
{
    return x + y + z;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    std::cout << "\nint add: " << add_num(1, 2, 3);
    std::cout << "\nlong add: " << add_num(11L, 2L, 3L);
    std::cout << "\nfloat add: " << add_num(1.1f, 2.2f, 3.f);
    std::cout << "\ndouble add: " << add_num(1.1, 2.2, 3.3);

    return 0;
}
