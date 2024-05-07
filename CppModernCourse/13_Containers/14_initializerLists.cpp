/**
 * @file 14_initializerLists.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <vector>

/**
 * @brief 可以在用户自定义类型中使用大括号初始化列表,
 * 具体通过 STL 的＜initializer_list＞头文件中的 std::initializer_list 容器来实现.
 * initializer_list 是一个类模板, 接受与大括号初始化列表中的基础类型相对应的单个模板参数. 
 * *该模板用作访问大括号初始化列表元素的简单代理.
 * 
 * initializer_list 是只读的,支持三种操作:
 * 1. size 方法返回 initializer_list 的元素数量;
 * 2. begin 和 end 方法返回通常的半开半闭区间迭代器;
 * ?通常, 应该设计函数按值接受 initializer_list 为参数;
 * 
 */

size_t square_root(size_t x)
{
    const auto result = static_cast<size_t>(sqrt(x));
    if (result * result != x)
        throw std::logic_error{"Not a perfect square."};
    return result;
}

template<typename T>
struct SquareMatrix
{
    SquareMatrix(std::initializer_list<T> val)
        : dim{square_root(val.size())}
        , data(dim, std::vector<T>{})
    {
        auto iter = val.begin();
        for (size_t row{0}; row < dim; ++row)
        {
            data[row].assign(iter, iter + dim);

            iter += dim;
        }
    }

    T &at(size_t row, size_t col)
    {
        if (row >= dim || col >= dim)
            throw std::out_of_range{"Index invalid\n"};

        return data[row][col];
    }

    const size_t dim;

private:
    std::vector<std::vector<T>> data;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nSquareMatrix and std::initializer_list\n");
    SquareMatrix<int> mat{1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    assert(mat.dim == 4);
    printf("The dimension of matrix is %lld\n", mat.dim);

    mat.at(1, 1) = 6;
    assert(mat.at(1, 1) == 6);
    printf("the value of position(mat.at(0, 2)): %d\n", mat.at(0, 2));

    return 0;
}
