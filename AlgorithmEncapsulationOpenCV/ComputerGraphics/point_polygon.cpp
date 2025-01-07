/**
 * @file point_polygon.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-07
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ point_polygon.cpp -std=c++20
 * clang++ point_polygon.cpp -std=c++20
 * 
 */

#include <cmath>
#include <iostream>
#include <vector>

#ifdef _WIN32
#    include <Windows.h>
#endif // _WIN32

/* 计算机图形学: 判断点是否在多边形内部
 * 在计算机图形学中, 判断点是否在多边形内部是一个常见的问题.
 * 介绍两种常用的方法: 射线法和回转数, 并提供相应的理论推导和代码实现.
 * 
 * 射线法(Ray Casting Method)的基本思想是从待判断的点出发,向任意方向发射一条射线,
 * 然后计算这条射线与多边形边界的交点数目;
 * 如果交点数为奇数,则该点在多边形内部; 如果为偶数, 则在外部.
 * 具体步骤如下:
 * 1. 选择一个方向, 从点P出发画一条射线;
 * 2. 遍历多边形的每条边, 检查射线是否与该边相交;
 * 3. 如果射线与边相交, 且交点在射线的延长线上(即交点的x坐标大于点P的x坐标), 则交点数加1;
 * 4. 遍历完所有边后, 如果交点数为奇数, 则点P在多边形内部; 如果为偶数, 则在外部;
 * 
 * 回转数算法(Winding Number Algorithm)是一种判断点是否在多边形内部的算法.
 * 回转数是指从点P出发,沿着多边形的边界走一圈,回到起点时,点P绕了多少圈;
 * 如果回转数为0,则点P在多边形外部; 如果回转数不为0, 则点P在多边形内部.
 * 具体步骤如下:
 * 1. 向量叉乘: 对于多边形的每条边, 计算从点P到边的两个端点的向量的叉乘,
 *    叉乘的符号可以告诉向量的旋转方向.
 * 2. 角度计算: 通过叉乘的符号, 可以计算出每条边对点P的回转贡献,
 *    如果叉乘为正, 表示逆时针旋转, 贡献为1; 如果叉乘为负, 表示顺时针旋转, 贡献为-1.
 * 3. 回转数累加: 遍历多边形的所有边, 累加每条边的回转贡献,
 *    如果点P在多边形内部, 累加的结果将是一个非零整数; 如果点P在多边形外部, 累加的结果将为0.
 * 
 */

struct Point
{
    double x;
    double y;
};

// 计算叉积，用于判断点P是否在A和B的连线上
double cross(const Point &A, const Point &B, const Point &C)
{
    return (B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x);
}

// 射线法, 判断点是否在多边形内部
bool isPointInPolygon(const std::vector<Point> &polygon, const Point &P)
{
    int wn = 0; // 交点数
    int n  = polygon.size();
    for (int i = 0; i < n; i++)
    {
        const Point &A = polygon[i];
        const Point &B = polygon[(i + 1) % n];
        if (A.y <= P.y && B.y > P.y && cross(B, P, A) > 0)
        {
            wn++;
        }
        else if (A.y > P.y && B.y <= P.y && cross(B, P, A) < 0)
        {
            wn--;
        }
    }
    return wn != 0; // 如果交点数不为0，则点在多边形内部
}

// 回转数算法, 判断点是否在多边形内部
int windingNumber(const std::vector<Point> &polygon, const Point &P)
{
    int wn = 0;
    for (size_t i = 0; i < polygon.size(); ++i)
    {
        Point A = polygon[i], B = polygon[(i + 1) % polygon.size()];
        if (A.y <= P.y && B.y > P.y)
        {
            if (cross(P, A, B) > 0)
            {
                ++wn;
            }
        }
        else if (A.y > P.y && B.y <= P.y)
        {
            if (cross(P, A, B) < 0)
            {
                --wn;
            }
        }
    }
    return wn;
}

bool isPointInPolygonWinding(const std::vector<Point> &polygon, const Point &P)
{
    return windingNumber(polygon, P) != 0;
}

// -------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif // _WIN32

    // 多边形顶点坐标
    std::vector<Point> polygon = {
        {0, 0},
        {2, 0},
        {1, 2}
    };
    // 待判断点的坐标
    Point P = {1, 1};

    // ================= 射线法
    // 判断点与多边形的位置关系
    if (isPointInPolygon(polygon, P))
    {
        std::cout << "[Ray-Casting]Point P is inside the polygon." << std::endl;
    }
    else
    {
        std::cout << "[Ray-Casting]Point P is outside the polygon." << std::endl;
    }

    // ================= 回转数算法
    std::vector<Point> polygon_ = {
        {0, 0},
        {1, 0},
        {1, 1},
        {0, 1}
    };
    Point P_ = {0.5, 1.5};
    if (isPointInPolygonWinding(polygon_, P_))
    {
        std::cout << "[Winding-Number]Point is inside the polygon." << std::endl;
    }
    else
    {
        std::cout << "[Winding-Number]Point is outside the polygon." << std::endl;
    }

    return 0;
}
