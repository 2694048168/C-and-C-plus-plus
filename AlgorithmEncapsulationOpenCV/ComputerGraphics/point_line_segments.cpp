/**
 * @file point_line_segments.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-07
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ point_line_segments.cpp -std=c++20
 * clang++ point_line_segments.cpp -std=c++20
 * 
 */

#include <cmath>
#include <iostream>

#ifdef _WIN32
#    include <Windows.h>
#endif // _WIN32

/* 计算机图形学：计算点与线段的距离、投影和位置关系
 * 在计算机图形学、几何计算和物理模拟等领域，经常需要计算一个点到线段的距离、投影和位置关系;
 * 这些计算可以帮助确定点是否在线段上、点到线段的最短距离、以及点相对于线段的位置等信息;
 * 详细介绍如何进行这些计算的理论推导, 并提供相应的代码实现.
 * Step1. 点到线段的距离, 给定P(x, y)和线段AB, 其中A(x1, y1)和B(x2, y2):
 *     - 首先计算向量 AP=(x - x1, y - y1) 和 AB=(x2 - x1, y2 - y1);
 *     - 点P到线段AB的距离通过向量叉乘的模除以向量AB的模 d = |AP X AB| / |AB|;
 *     - 叉乘的模 |AP X AB| = |(x - x1)(y2 - y1) - (y - y1)(x2 - x1)|;
 *     - 向量AB的模 |AB| = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
 * 
 * Step2. 点到线段的投影, 点P到线段AB的投影点P'可以通过向量投影的方法计算:
 *     - 首先计算点 P在向量AB 上的(点乘)投影长度 λ=AP . AB / pow(|AB|, 2);
 *     - 然后根据 λ 的值确定投影点P'的位置:
 *         - λ∈[0,1], P' 在线段 AB 上;
 *         - λ<0, P' 在线段 BA 的延长线上;
 *         - λ>1, P' 在线段 AB 的延长线上;
 *         - P'(x1 + λ(x2 - x1), y1 + λ(y2 - y1));
 * 
 * Step3. 点与线段的位置关系,通过判断AP和AB 叉乘的符号,确定点P相对于线段AB的位置关系:
 *     - AP X AB > 0, 点P在线程AB的左侧;
 *     - AP X AB < 0, 点P在线程AB的右侧;
 *     - AP X AB = 0, 点P在线程AB上;
 *     - 判断标准基于右手定则,
 * 即如果你的右手的拇指指向AB的方向, 而食指指向AP的方向, 那么你的中指将指向叉积的方向;
 * 如果中指指向屏幕外(或向上的方向), 则P在AB的左侧；
 * 如果中指指向屏幕内(或向下的方向), 则P在AB的右侧.
 * 
 */

struct Point
{
    float x;
    float y;
};

Point operator-(const Point &a, const Point &b)
{
    return {a.x - b.x, a.y - b.y};
}

float dotProduct(const Point &a, const Point &b)
{
    return a.x * b.x + a.y * b.y;
}

float crossProduct(const Point &a, const Point &b)
{
    return a.x * b.y - a.y * b.x;
}

float magnitude(const Point &a)
{
    return std::sqrt(a.x * a.x + a.y * a.y);
}

float pointToSegmentDistance(const Point &A, const Point &B, const Point &P)
{
    Point AP          = P - A;
    Point AB          = B - A;
    float cross       = std::abs(crossProduct(AP, AB));
    float abMagnitude = magnitude(AB);
    return cross / abMagnitude;
}

Point pointToSegmentProjection(const Point &A, const Point &B, const Point &P)
{
    Point AP        = P - A;
    Point AB        = B - A;
    float dot       = dotProduct(AP, AB);
    float abSquared = AB.x * AB.x + AB.y * AB.y;
    float lambda    = dot / abSquared;
    Point PPrime    = {A.x + lambda * AB.x, A.y + lambda * AB.y};
    return PPrime;
}

void determinePosition(const Point &A, const Point &B, const Point &P)
{
    Point AP    = P - A;
    Point AB    = B - A;
    float cross = crossProduct(AB, AP);

    if (cross > 0)
    {
        std::cout << "点P在线段AB的左侧" << std::endl;
    }
    else if (cross < 0)
    {
        std::cout << "点P在线段AB的右侧" << std::endl;
    }
    else
    {
        std::cout << "点P在直线AB上" << std::endl;
    }
}

// -------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

    Point A = {1, 2};
    Point B = {4, 6};
    Point P = {3, 4};

    std::cout << "点P到线段AB的距离: " << pointToSegmentDistance(A, B, P) << std::endl;

    Point PPrime = pointToSegmentProjection(A, B, P);
    std::cout << "点P到线段AB的投影点: (" << PPrime.x << ", " << PPrime.y << ")" << std::endl;

    determinePosition(A, B, P);

    return 0;
}
