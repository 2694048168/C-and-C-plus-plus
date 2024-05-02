/**
 * @file 05_stdFunction.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstdio>
#include <functional>

/**
 * @brief std::function 
 * ! function 类在 STL 中
 * 只需要一个统一的容器来存储可调用对象(callable object), 
 * ＜functional＞头文件中的std::function 类模板是可调用对象的多态包装器,
 * 换句话说, 是一个通用的函数指针(generic function pointer), 
 * 可以将静态函数, 函数对象或lambda存储到std::function.
 * 
 * 使用 function:
 * 1. 在调用者不知道函数实现的情况下调用;
 * 2. 赋值、移动和复制;
 * 3. 有空状态，类似于 nullptr;
 * 
 */

void static_func()
{
    printf("A static function.\n");
}

// 函数类型, 计算以空字符(null)结尾的字符串中特定字符出现的频率
struct CountIf
{
    CountIf(char x)
        : x{x}
    {
    }

    size_t operator()(const char *str) const
    {
        size_t index{}, result{};
        while (str[index])
        {
            if (str[index] == x)
                result++;
            index++;
        }
        return result;
    }

private:
    const char x;
};

size_t count_spaces(const char *str)
{
    size_t index{}, result{};
    while (str[index])
    {
        if (str[index] == ' ')
            result++;
        index++;
    }
    return result;
}

// ----------------------------------
int main(int argc, const char **argv)
{
    // Declaring a Function
    printf("===== step 1. Empty Functions\n");
    std::function<void()> func;
    try
    {
        // std::function 将抛出 std::bad_function_call 异常
        func();
    }
    catch (const std::bad_function_call &e)
    {
        printf("Exception: %s\n", e.what());
    }

    // 将可调用对象赋值给函数
    printf("===== step 2. Assigning a Callable Object to a Function\n");
    std::function<void()> func_2{[]
                                 {
                                     printf("A lambda.\n");
                                 }};
    func_2();
    func_2 = static_func;
    func_2();

    /**
     * @brief 可以使用可调用对象构造函数, 只要该对象支持函数模板参数所隐含的函数语义
     * 
     */
    std::function<size_t(const char *)> funcs[]{count_spaces, CountIf{'e'}, [](const char *str)
                                                {
                                                    size_t index{};
                                                    while (str[index]) index++;
                                                    return index;
                                                }};

    auto   text = "Sailor went to sea to see what he could see.";
    size_t index{};
    // 请注意 从 main 的角度来看, funcs 中的所有元素都是相同的:
    // 只需使用一个以空字符结尾的字符串来调用它们并返回 size_t.
    for (const auto &func : funcs)
    {
        printf("func #%zd: %zd\n", index++, func(text));
    }

    /**
     * @brief 使用 function 会产生运行时开销,
     * 由于技术原因, function 可能需要进行动态分配来存储可调用对象;
     * 编译器也很难优化掉 function 调用, 所以经常会遇到间接函数调用,
     * 间接函数调用需要额外的指针解引用.
     */

    return 0;
}
