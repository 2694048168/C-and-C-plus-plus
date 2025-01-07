/**
 * @file intersection_line_segments.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-07
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ intersection_line_segments.cpp -std=c++20
 * clang++ intersection_line_segments.cpp -std=c++20
 * 
 */

#include <iostream>
#include <limits>
#include <string>

/* 计算机图形学: 线段的交点计算及实现
 * 在计算机图形学、几何计算和工程领域中, 经常需要计算两条线段的交点;
 * 详细介绍线段交点计算的原理、公式推导过程以及代码实现.
 * Step1. 要计算两条线段的交点, 首先需要确定这两条线段是否相交;
 * Step2. 线段可以由两个端点定义, 设线段AB的端点为A(x1, y1)和B(x2, y2);线段CD的端点为C(x3, y3)和D(x4, y4);
 * Step3. 确定线段的方程: 线段(AB)和(CD)可以表示为参数方程; 斜率 k1∈[0,1] and k2∈[0,1];
 * Step4. 设置交点条件: 如果线段相交, 那么存在和使得; 必定存在同一点P(x0, y0),使得两个直线方程成立;
 * Step5. 解方程组: 解这个方程组得到 k1 and k2; k1∈[0,1] and k2∈[0,1] 同时满足则必定相交;
 * Step6. 计算交点: 交点的坐标为P(x1 + k1(x2 - x1), y1 + k1(y2 - y1)) = P(x0, y0);
 * 
 * @note: This is tricky. Most Floating Point (FP) operations are rounded,
 *   which introduces an error. If x=a op b, then the error is 
 *   err(x) <= 0.5 * x * std::numeric_limits<typeof(x)>::epsilon. 
 * On some FP rounding modes don't multiply by 0.5.
 * 
 */

[[nodiscard]] inline static bool approximatelyEqual(float a, float b,
                                                    float epsilon = std::numeric_limits<float>::epsilon()) noexcept
{
    return fabs(a - b) <= ((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

[[nodiscard]] inline static bool essentiallyEqual(float a, float b,
                                                  float epsilon = std::numeric_limits<float>::epsilon()) noexcept
{
    return fabs(a - b) <= ((fabs(a) > fabs(b) ? fabs(b) : fabs(a)) * epsilon);
}

struct Point
{
    float x;
    float y;
};

struct LineSegments
{
    Point startPoint;
    Point endPoint;
};

Point line_intersection(Point A, Point B, Point C, Point D)
{
    float x1 = A.x, y1 = A.y;
    float x2 = B.x, y2 = B.y;
    float x3 = C.x, y3 = C.y;
    float x4 = D.x, y4 = D.y;

    // 计算分母
    float denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
    if (essentiallyEqual(denominator, 0))
    {
        return {0.f, 0.f}; // 线段平行或重合
    }

    // 计算 t 和 s
    float t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator;
    float s = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denominator;

    // 检查 t 和 s 是否在 [0, 1] 区间内
    if (0 <= t && t <= 1 && 0 <= s && s <= 1)
    {
        return {x1 + t * (x2 - x1), y1 + t * (y2 - y1)};
    }
    else
    {
        return {0.f, 0.f}; // 线段不相交
    }
}

Point line_intersection(LineSegments AB, LineSegments CD)
{
    float x1 = AB.startPoint.x, y1 = AB.startPoint.y;
    float x2 = AB.startPoint.x, y2 = AB.endPoint.y;
    float x3 = CD.startPoint.x, y3 = CD.startPoint.y;
    float x4 = CD.endPoint.x, y4 = CD.endPoint.y;

    // 计算分母
    float denominator = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3);
    if (essentiallyEqual(denominator, 0))
    {
        return {0.f, 0.f}; // 线段平行或重合
    }

    // 计算 t 和 s
    float t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator;
    float s = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / denominator;

    // 检查 t 和 s 是否在 [0, 1] 区间内
    if (0 <= t && t <= 1 && 0 <= s && s <= 1)
    {
        return {x1 + t * (x2 - x1), y1 + t * (y2 - y1)};
    }
    else
    {
        return {0.f, 0.f}; // 线段不相交
    }
}

// -------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    Point A{0.3f, 0.3f};
    Point B{0.8f, 0.8f};
    Point C{0.2f, 0.8f};
    Point D{0.8f, 0.2f};

    auto ret = line_intersection(A, B, C, D);
    std::cout << "The two line segments intersection as Point: { " << ret.x << ", " << ret.y << "}\n";

    // --------------------------------------------------
    // ((0.2, 0.2), (0.8, 0.2), (0.5, 0.5), (0.8, 0.5)),
    LineSegments AB{
        {0.2f, 0.2f},
        {0.8f, 0.2f}
    };
    LineSegments CD{
        {0.5f, 0.5f},
        {0.8f, 0.5f}
    };

    ret = line_intersection(AB, CD);
    std::cout << "The two line segments intersection as Point: { " << ret.x << ", " << ret.y << "}\n";

    return 0;
}
