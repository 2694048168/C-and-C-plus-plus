/**
 * @file 07_NonTypeTemplateParameters.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstddef>
#include <cstdio>
#include <stdexcept>

/**
 * @brief Non-Type Template Parameters 非类型模板参数
 * 用 typename（或 class）关键字声明的模板参数称为类型模板参数,
 *  它是一些尚未指定的类型的替身; 也可以使用非类型模板参数, 是一些尚未指定的值的替身.
 * 非类型模板参数可以是以下任意一种:
 * 1. 整数型;
 * 2. 左值引用类型;
 * 3. 指针(或成员指针)类型;
 * 4. std::nullptr_t(nullptr 的类型);
 * 5. enum class;
 * 
 * 使用非类型模板参数可以在编译时向泛型代码中注入值,
 * 例如可以构造一个名为 get 的模板函数, 
 * 通过将要访问的索引作为非类型模板参数可以在编译时检查是否有越界数组访问.
 * 
 */
int &get_int(int (&arr)[10], size_t index)
{
    if (index >= 10)
        throw std::out_of_range{"Out of bounds"};
    return arr[index];
}

template<size_t index, typename T, size_t Length>
T &get(T (&arr)[Length])
{
    static_assert(index < Length, "Out-of-bounds access\n");
    return arr[index];
}

// ----------------------------------
int main(int argc, const char **argv)
{
    int fib[]{1, 1, 2, 0};

    printf("%d %d %d ", get<0>(fib), get<1>(fib), get<2>(fib));
    get<3>(fib) = get<1>(fib) + get<2>(fib);
    printf("%d", get<3>(fib));

    // printf("%d", get<4>(fib)); // !error

    return 0;
}
