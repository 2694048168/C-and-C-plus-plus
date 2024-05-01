/**
 * @file 05_constantExpressions.cpp
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

/**
 * @brief 常量表达式 Constant Expressions
 * 常量表达式是可以在编译时求值的表达式, 出于性能和安全性考虑
 * 
 * 表达式 constexpr 扩展编译器的作用范围,
 * 只要在编译时存在计算表达式所需的所有信息, 并将该表达式标记为 constexpr,
 * 编译器就将被迫这样做, 这种简单的承诺会对代码的可读性和运行时性能产生巨大影响.
 * 
 * ---- const 和 constexpr 密切相关
 * constexpr 强制表达式在编译时求值, 而 const则强制变量在某些作用域（运行时）内不可更改,
 * 所有的 constexpr 表达式都是 const, 因为它们在运行时总是固定的.
 * 
 */

//  Color 对象是用"色相-饱和度-值"(Hue-Saturation-Value, HSV)编码的
struct Color
{
    float H;
    float S;
    float V;
};

// "红-绿-蓝"(Red-Green-Blue,RGB)表示实例化 Color 对象
// 通过转换器手动将 RGB 转换为 HSV, 这是一个很好的使用 constexpr 消除魔数的示例
constexpr uint8_t max(uint8_t a, uint8_t b)
{
    return a > b ? a : b;
}

constexpr uint8_t max(uint8_t a, uint8_t b, uint8_t c)
{
    return max(max(a, b), max(a, c));
}

constexpr uint8_t min(uint8_t a, uint8_t b)
{
    return a < b ? a : b;
}

constexpr uint8_t min(uint8_t a, uint8_t b, uint8_t c)
{
    return min(min(a, b), min(a, c));
}

constexpr float modulo(float dividend, float divisor)
{
    const auto quotient = dividend / divisor;
    return divisor * (quotient - static_cast<uint8_t>(quotient));
}

/**
 * @brief 声明并初始化 Color 对象c, 它将最终由 rgb_to_hsv 函数返回,
 * Color 的值 V 是通过按比例缩放 r、g 和 b 的最大值来计算的;
 * 接下来, 通过计算最小和最大 RGB 值之间的距离并按 V 缩放来计算饱和度 S;
 * 如果将 HSV 值想象成存在于圆柱体内, 则饱和度是沿水平轴的距离, 值是沿垂直轴的距离, 色相是角度;
 * 本质上, 它需要将角度计算为相对于主要颜色成分的角度的偏移量, 
 * 将其缩放并进行模化以适合 0 到 360°的区间, 并存储到 H 中.
 * 
 */
constexpr Color rgb_to_hsv(uint8_t r, uint8_t g, uint8_t b)
{
    Color      c{};
    const auto c_max = max(r, g, b);
    c.V              = c_max / 255.0f;
    const auto c_min = min(r, g, b);
    const auto delta = c.V - c_min / 255.0f;
    c.S              = c_max == 0 ? 0 : delta / c.V;
    if (c_max == c_min)
    {
        c.H = 0;
        return c;
    }
    if (c_max == r)
    {
        c.H = (g / 255.0f - b / 255.0f) / delta;
    }
    else if (c_max == g)
    {
        c.H = (b / 255.0f - r / 255.0f) / delta + 2.0f;
    }
    else if (c_max == b)
    {
        c.H = (r / 255.0f - g / 255.0f) / delta + 4.0f;
    }
    c.H *= 60.0f;
    c.H = c.H >= 0.0f ? c.H : c.H + 360.0f;
    c.H = modulo(c.H, 360.0f);
    return c;
}

// -----------------------------------
int main(int argc, const char **argv)
{
    constexpr auto val = 3.1415926 * 120 * 42;

    printf("the const value: %f\n", val);
    // * @note https://godbolt.org/z/Wra1rj51r

    printf("========= constexpr example =========\n");
    auto black = rgb_to_hsv(0, 0, 0);
    printf("the HSV color value: (%f, %f, %f)\n", black.H, black.S, black.V);

    auto white = rgb_to_hsv(255, 255, 255);
    printf("the HSV color value: (%f, %f, %f)\n", white.H, white.S, white.V);

    auto red = rgb_to_hsv(255, 0, 0);
    printf("the HSV color value: (%f, %f, %f)\n", red.H, red.S, red.V);

    auto green = rgb_to_hsv(0, 255, 0);
    printf("the HSV color value: (%f, %f, %f)\n", green.H, green.S, green.V);

    auto blue = rgb_to_hsv(0, 0, 255);
    printf("the HSV color value: (%f, %f, %f)\n", blue.H, blue.S, blue.V);

    return 0;
}
