/**
 * @file 04_lambdaExpressions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-02
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <cstddef>
#include <cstdio>

/**
 * @brief Lambda Expressions lambda 表达式
 * lambda 表达式简洁地构造未命名的函数对象, 其函数对象隐含了函数类型,
 * 因而可以快速地动态声明函数对象,当只需要在单个上下文中初始化函数对象时,使用就非常方便.
 * lambda 表达式有五个组件:
 * [captures] (parameters) modifiers -＞ return-type❹ { body }
 * 1. 捕获列表 captures: 函数对象的成员变量, 即局部应用的参数;
 * 2. 参数 parameters: 调用函数对象所需的参数;
 * 3. 表达式体 body: 函数对象的代码;
 * 4. 修饰符 modifiers: constexpr、mutable、noexcept 和 [[noreturn]] 等元素;
 * 5. 返回类型 return-type: 函数对象返回的类型.
 * 
 * ---- 参数和表达式体 Lambda Parameters and Bodies
 * ---- 默认参数 Default Arguments
 * ---- 泛型 Generic Lambdas
 * ---- 编译器会自动推断 lambda 的返回类型, 可以使用箭头后缀方式 -＞显式指定
 * ---- 捕获列表(Lambda Captures), 包含任意数量的以逗号分隔的参数
 * *lambda 表达式可以按引用也可以按值给出捕获列表, 默认情况下采取按值捕获.
 * *默认捕获列表 [&], [=], 
 * 
 * ---- 捕获列表中的初始化表达式 Initializer Expressions in Capture Lists
 * 捕获列表中使用初始化表达式的方式也称为初始化捕获
 * ---- this 捕获
 * lambda 表达式有封闭类(enclosing class),
 *  使用 [*this] 通过按值捕获方式捕获封闭对象(this 指向的对象).
 *  使用 [this] 通过按引用捕获的方式捕获封闭对象(this 指向的对象).
 * 
 * 
 * 
 * 
 */
template<typename Func>
void transform(Func func, const int *in, int *out, size_t length)
{
    for (size_t i{0}; i < length; ++i)
    {
        out[i] = func(in[i]);
    }
}

template<typename Func, typename DataType>
void transform(Func func, const DataType *in, DataType *out, size_t len)
{
    for (size_t i{}; i < len; ++i)
    {
        out[i] = func(in[i]);
    }
}

class LambdaFactory
{
public:
    LambdaFactory(char in)
        : to_count{in}
        , tally{}
    {
    }

    auto make_lambda()
    {
        return [this](const char *str)
        {
            size_t index{}, result{};
            while (str[index])
            {
                if (str[index] == to_count)
                    result++;
                index++;
            }
            tally += result;
            return result;
        };
    }

public:
    const char to_count;
    size_t     tally;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    const size_t len{3};

    int base[]{1, 2, 3};
    int a[len];
    int b[len];
    int c[len];
    transform([](int x) { return 1; }, base, a, len);
    transform([](int x) { return x; }, base, b, len);
    transform([](int x) { return 10 * x + 5; }, base, c, len);

    for (size_t i{}; i < len; i++)
    {
        printf("Element %zd: %d %d %d\n", i, a[i], b[i], c[i]);
    }

    printf("============= Default Arguments ==============\n");
    auto increment = [](auto x, int y = 1u)
    {
        return x + y;
    };
    printf("increment(10) = %d\n", increment(10));
    printf("increment(10, 5) = %d\n", increment(10, 5));

    printf("============= Generic Lambdas ==============\n");
    int   base_int[]{1, 2, 3};
    int   a_base[len];
    float base_float[]{10.f, 20.f, 30.f};
    float b_base[len];

    /* Generic Lambdas */
    auto translate = [](auto x)
    {
        return 10 * x + 5;
    };

    transform(translate, base_int, a_base, len);
    transform(translate, base_float, b_base, len);
    for (size_t i{}; i < len; i++)
    {
        printf("Element %zd: %d %f\n", i, a_base[i], b_base[i]);
    }

    /**
     * @brief mutable 关键字添加到 lambda 表达式中,
     * 否则不允许修改按值捕获的变量, mutable 关键字允许修改按值捕获的变量,
     * 这包括在该对象上调用非 const 方法.
     */
    printf("============= mutable ==============\n");
    char   to_count{'s'};
    size_t tally{};

    auto s_counter = [=](const char *str) mutable
    {
        size_t index{}, result{};
        while (str[index])
        {
            if (str[index] == to_count)
                result++;
            index++;
        }
        tally += result;
        return result;
    };
    auto sally = s_counter("Sally sells seashells by the seashore.");
    printf("Tally: %zd\n", tally);
    printf("Sally: %zd\n", sally);
    printf("Tally: %zd\n", tally);

    auto sailor = s_counter("Sailor went to sea to see what he could see.");
    printf("Sailor: %zd\n", sailor);
    printf("Tally: %zd\n", tally);

    printf("============= Capturing this ==============\n");
    LambdaFactory factory{'s'};

    auto lambda = factory.make_lambda();
    printf("Tally: %zd\n", factory.tally);
    printf("Sally: %zd\n", lambda("Sally sells seashells by the seashore."));
    printf("Tally: %zd\n", factory.tally);
    printf("Sailor: %zd\n", lambda("Sailor went to sea to see what he could see."));
    printf("Tally: %zd\n", factory.tally);

    /**
     * @brief 
     * 捕获列表   |    含义
     * [&]            按引用默认捕获
     * [&,i]          按引用默认捕获；按值捕获 i
     * [=]            按值默认捕获
     * [=,&i]         按值默认捕获；按引用捕获 i
     * [i]            按值捕获 i
     * [&i]           按引用捕获 i
     * [i,&j]         按值捕获 i；按引用捕获 j
     * [i=j,&k]       按值捕获 j 并将其作为 i；按引用捕获 k
     * [this]         按引用捕获封闭对象
     * [*this]         按值捕获封闭对象
     * [=,*this,i,&j]  按值默认捕获；按值捕获 this 和 i；按引用捕获 j
     * 
     * TODO: constexpr lambda 表达式
     * TODO: 标准委员会计划在每次发布新版本时放宽这些限制,
     * 因此如果要使用constexpr 编写大量代码, 请务必了解最新的 constexpr 限制.
     */

    return 0;
}
