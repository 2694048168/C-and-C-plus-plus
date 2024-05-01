/**
 * @file 04_typeConversions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-01
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdint>
#include <cstdio>
#include <limits>

/**
 * @brief 类型转换 Type Conversions
 * 将一种类型转换为另一种类型时, 可以执行类型转换,
 * *根据情况, 类型转换可以是显式的(explicit), 也可以是隐式的(implicit).
 * 
 * ==== Implicit Type Conversions 隐式类型转换
 * 隐式类型转换可以在需要使用特定类型但却提供了一个其他类型的任何地方进行.
 * 这些转换发生在不同的上下文中, 每当发生算术运算时,
 * 较短的整数类型都会被提升为 int 类型; 在算术运算期间, 整数类型也可以提升为浮点类型.
 * 
 * !1. 浮点类型到整数类型的转换
 * !2. 整数类型到整数类型的转换
 * !3. 浮点类型到浮点类型的转换
 * !4. 转换为 bool 类型
 * !5. 指针转换为 void*
 * 
 * ==== Explicit Type Conversion 显式类型转换
 * 显式类型转换(explicit type conversion)简称为类型转换(cast)
 * 1. 使用大括号初始化(braced initialization {}), 
 *    主要优点是对所有类型都安全且不会发生窄化(non-narrowing)
 * 2. C 风格的类型转换 (desired-type)object-to-cast
 *   针对每种 C 风格的类型转换,对应的 static_cast,const_cast,reinterpret_cast
 * 3. 用户自定义类型的转换
 *  对于用户自定义类型, 可以提供用户自定义类型转换功能,
 *  这些函数会告诉编译器在隐式和显式类型转换期间用户自定义类型的行为.
 * 
 */

void print_addr(void *x)
{
    printf("0x%p\n", x);
}

// C 风格类型转换的“火车事故”—意外地丢掉了 read_only 的 const限定符
void trainwreck(const char *read_only)
{
    auto as_unsigned = (unsigned char *)read_only;
    // auto as_unsigned = reinterpret_cast<unsigned char *>(read_only); //!ERROR
    *as_unsigned = 'b'; // Crashes on Windows 10 x64
}

// 隐式转换
struct ReadOnlyInt
{
    ReadOnlyInt(int val)
        : val{val}
    {
    }

    // 隐式转换
    operator int() const
    {
        return val;
    }

private:
    const int val;
};

// 使用 explicit 关键字可以实现显式转换,
// 显式构造函数指示编译器不要将构造函数当作隐式转换的一种方式
struct ReadOnlyIntExplicit
{
    ReadOnlyIntExplicit(int val)
        : val{val}
    {
    }

    // 显式转换
    explicit operator int() const
    {
        return val;
    }

private:
    const int val;
};

// -----------------------------------
int main(int argc, const char **argv)
{
    auto    x = 2.7182818284590452353602874713527L;
    uint8_t y = x; // Silent truncation
    printf("the value of x: %Lf, and y: %d\n", x, y);

    uint8_t y_{x}; // !ERROR Bang!
    printf("the value of x: %Lf, and y_: %d\n", x, y_);

    // 0b111111111 = 511
    uint8_t x_val = 0b111111111; // 255
    int8_t  y_val = 0b111111111; // Implementation defined.
    printf("x_val: %u, y_val: %d\n", x_val, y_val);

    double      x_float       = std::numeric_limits<float>::max();
    long double y_double      = std::numeric_limits<double>::max();
    float       z_long_double = std::numeric_limits<long double>::max(); // Undefined Behavior
    printf("x_float: %g, y_double: %Lg, z_long_double: %g\n", x_float, y_double, z_long_double);

    // 指针始终可以隐式转换为 void*
    int x_ptr{};
    print_addr(&x_ptr);
    print_addr(nullptr);

    // 使用大括号初始化可确保在编译时只有安全、行为良好且非窄化的转换发生
    int32_t a = 100;
    int64_t b{a};
    if (a == b)
        printf("Non-narrowing conversion!\n");
    // int32_t c{ b }; // !Bang!

    // ==========User-Defined Type Conversions
    ReadOnlyInt the_answer{42};
    auto        ten_answers = the_answer * 10; // int with value 420
    printf("the value of ten_answers: %d\n", ten_answers);

    ReadOnlyIntExplicit the_answer_{42};
    auto                ten_answers_ = static_cast<int>(the_answer_) * 10;
    printf("the value of ten_answers_: %d\n", ten_answers_);

    // C-Style Casts
    auto ezra = "Ezra";
    printf("Before trainwreck: %s\n", ezra);
    trainwreck(ezra);
    printf("After trainwreck: %s\n", ezra);
    // 现代操作系统会严格限制内存访问模式,
    // 尝试写入内存, 该内存存储了字符串字面量 Ezra,
    // 在 Windows 10 x64 计算机上, 程序会因内存访问冲突（这是只读内存）而崩溃.

    return 0;
}
