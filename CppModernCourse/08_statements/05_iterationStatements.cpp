/**
 * @file 05_iterationStatements.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdint>
#include <cstdio>

/**
 * @brief Iteration Statements 迭代语句
 * 迭代语句重复执行一条语句, 有四种迭代语句:
 * 1. while 循环 | The while loop
 * 2. do-while 循环 | The do-while loop
 * 3. for 循环 | The for loop
 * 4. 基于范围的 for 循环 | The range-based for loop
 * 
 * ====== 在涉及迭代的大多数情况下, 需要执行以下三个任务:
 * 1）初始化一些对象;
 * 2）在每次迭代之前更新对象;
 * 3）检查对象的值是否满足某些条件.
 * 
 */

bool double_return_overflow(uint8_t &x)
{
    const auto original = x;
    x *= 2;
    return original > x;
}

struct FibonacciIterator
{
    bool operator!=(int x) const
    {
        return x >= current;
    }

    FibonacciIterator &operator++()
    {
        const auto tmp = current;
        current += last;
        last = tmp;
        return *this;
    }

    int operator*() const
    {
        return current;
    }

private:
    int current{1}, last{1};
};

struct FibonacciRange
{
    explicit FibonacciRange(int max)
        : max{max}
    {
    }

    FibonacciIterator begin() const
    {
        return FibonacciIterator{};
    }

    int end() const
    {
        return max;
    }

private:
    const int max;
};

// ------------------------------------
int main(int argc, const char **argv)
{
    printf("============== The while loop ==============\n");
    uint8_t x{1};
    printf("uint8_t:\n===\n");
    while (!double_return_overflow(x))
    {
        printf("%u ", x);
    }

    printf("\n============== The do-while loop ==============\n");
    uint8_t x_{1};
    printf("uint8_t:\n===\n");
    do
    {
        printf("%u ", x_);
    }
    while (!double_return_overflow(x_));

    printf("\n============== The for loop ==============\n");
    //  * @brief 初始化表达式、条件表达式和迭代表达式
    //  * @brief initialization, conditional, and iteration
    const int x_arr[]{1, 1, 2, 3, 5, 8}; /* the first six Fibonacci numbers */
    printf("i: x[i]\n");
    for (int idx{}; idx < 6; ++idx) // Iterating with an Index
    {
        printf("%d: %d\n", idx, x_arr[idx]);
    }

    printf("\n============== The range-based for loop ==============\n");
    for (const auto element : x_arr)
    {
        printf("%d ", element);
    }

    /**
     * @brief 范围表达式 Range Expressions
     * 可以定义自己的类型, 这些类型也是有效的范围表达式, 但是需要在类型上指定几个函数;
     * 每个范围都公开一个 begin 和 end 方法, 这些函数是公共接口,
     * 基于范围的 for 循环使用该公共接口与范围进行交互, 两个方法都返回迭代器,
     * ?迭代器是支持 operator!=, operator++ 和 operator* 的对象.
     *
     * !begin 和 end 返回的类型不必相同, 要求是 begin 中的 operator!= 
     * !接受一个 end 参数以支持比较运算 begin != end.
     * 
     * 斐波那契数范围 A Fibonacci Range
     */
    printf("\n============== A Fibonacci Range ==============\n");
    for (const auto i : FibonacciRange{100})
    {
        printf("%d ", i);
    }

    printf("\n============== A Fibonacci Range ==============\n");
    FibonacciRange range{500};
    const auto     end = range.end();
    for (auto x = range.begin(); x != end; ++x)
    {
        const auto i = *x;
        printf("%d ", i);
    }

    /**
     * @brief 跳转语句 Jump Statements
     * 跳转语句(break, continue, goto)转移控制流, 跳转语句没有条件, 应该避免使用
     * * break 语句终止封闭迭代或 switch 语句的执行;
     * * continue 语句会跳过封闭迭代语句的其余部分, 继续进行下一个迭代;
     * * goto 语句是无条件跳转语句, goto 语句的目标是标签;
     * 
     * NOTE: try-catch 块也是语句.
     */

    return 0;
}
