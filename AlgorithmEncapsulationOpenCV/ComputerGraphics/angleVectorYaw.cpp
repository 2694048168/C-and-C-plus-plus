/**
 * @file angleVectorYaw.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 计算两个向量Yaw的夹角
 * @version 0.1
 * @date 2025-02-09
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ angleVectorYaw.cpp -std=c++20
 * clang++ angleVectorYaw.cpp -std=c++20 -finput-charset=utf-8 -fexec-charset=utf-8
 * 
 */

#include <cmath>
// #define _USE_MATH_DEFINS
// /* MSVC compiler 需要该宏开启 M_PI 等数学常量可以在全局命名空间中使用 */
#include <corecrt_math_defines.h>
#include <iostream>

/**
  * @brief 在计算两个向量的夹角时, 通常使用点积和叉积来分别确定夹角的大小和方向
  * 1. 点积(Dot Product) 用于计算夹角的大小: A.B=|A| |B| cos(theta);
  *    通过点积可以得到夹角的余弦值, 从而计算出夹角的大小
  * 2. 叉积(Cross Product) 用于确定夹角的方向, AXB=|A| |B| sin(theta);
  *    在二维平面上, 叉积是一个标量, 表示两个向量的旋转方向;
  *    如果叉积为正,表示从向量 A 到向量 B 是逆时针旋转; 如果叉积为负,则是顺时针旋转;
  * 3. std::atan2(y, x): 一个计算反正切的函数,根据两个参数 y 和 x 的符号来确定角度所在的象限;
  *   它的返回值范围是 [-π, π],即 [-180°, 180°];
  *   这个函数通常用于计算一个向量与 x 轴正方向的夹角.
  * 
  */

// 将角度转换为弧度
double degreesToRadians(double degrees)
{
    return degrees * M_PI / 180.0;
}

// 将弧度转换为角度
double radiansToDegrees(double radians)
{
    return radians * 180.0 / M_PI;
}

// 计算两个向量的夹角（考虑正负方向）
double angleDifferenceWithSign(double yaw1, double yaw2)
{
    // 将yaw角转换为弧度
    double yaw1_rad = degreesToRadians(yaw1);
    double yaw2_rad = degreesToRadians(yaw2);

    // 将yaw角转换为二维向量
    double x1 = std::cos(yaw1_rad), y1 = std::sin(yaw1_rad);
    double x2 = std::cos(yaw2_rad), y2 = std::sin(yaw2_rad);

    // 计算夹角的正弦和余弦值
    double sin_theta = x1 * y2 - y1 * x2; // 叉积
    double cos_theta = x1 * x2 + y1 * y2; // 点积

    // 使用atan2计算夹角（考虑正负方向）
    double angle_rad = std::atan2(sin_theta, cos_theta);

    // 将弧度转换为角度
    double angle_deg = radiansToDegrees(angle_rad);

    return angle_deg;
}

// --------------------------------------
int main(int argc, const char *argv[])
{
    double yaw1, yaw2;
    std::cout << "Please input first-yaw angle(degree): ";
    std::cin >> yaw1;
    std::cout << "Please input second-yaw angle(degree): ";
    std::cin >> yaw2;

    double diff = angleDifferenceWithSign(yaw1, yaw2);
    std::cout << "Tow-yaw angle diff: " << diff << " degree" << std::endl;

    return 0;
}
