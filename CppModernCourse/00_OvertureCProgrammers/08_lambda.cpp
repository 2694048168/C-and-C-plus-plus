/**
 * @file 08_lambda.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-17
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <vector>

/**
 * @brief lambda, 也被称为无名函数或匿名函数, 是C++一个强大的语言特性, 可以提高代码的局部性;
 *  在某些情况下, 应该将指针传递给函数, 使用函数指针作为新创建的线程的目标函数,
 *  或者对序列的每个元素进行一些转换, 一般来说, 定义一个一次性使用的自由函数通常很不方便,
 *  这就是lambda发挥作用的地方; lambda是一个新的、自定义的函数, 与调用的其他参数一起内联定义;
 * [capture] (arguments) { body }
 * 
 * std:.count_if 调用中, lambda不需要捕获任何变量, 它所需要的所有信息就是单一参数 number;
 * 因为编译器知道 x 中包含的元素的类型, 所以用 auto 声明 number 的类型,
 * 这样编译器就可以自己推导出来; lambda被调用时, x中的每个元素都作为 number 参数传入;
 * 在 body 中, 只有当数字能被2整除时, 该lambda返回 true, 所以只有为偶数时才会计数;
 * 
 * C语言中不存在 lambda, 也不可能真正模拟它; 每次需要函数对象时, 都需要声明一个单独的函数
 * 而且不可能以同样的方式将对象捕获到一个函数中.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    std::vector<int> vec{0, 1, 8, 12, 5, 2, 3};

    auto num_evens = std::count_if(vec.cbegin(), vec.cend(), [](auto number) { return number % 2 == 0; });

    for (const auto &elem : vec)
    {
        std::cout << elem << ' ';
    }
    std::cout << "\n========== number of even: ";
    std::cout << num_evens << '\n';

    return 0;
}
