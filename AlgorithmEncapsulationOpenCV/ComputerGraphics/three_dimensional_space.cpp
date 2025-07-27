/**
 * @file three_dimensional_space.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 三维空间中的向量与坐标系变换
 * @version 0.1
 * @date 2025-07-27
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ three_dimensional_space.cpp -std=c++20
 * clang++ three_dimensional_space.cpp -std=c++20
 * 
 */

#include <array>
#define _USE_MATH_DEFINES // Required for some compilers
#include <cmath>
#include <iostream>
#include <numbers> // For C++20 and later

// 定义三维向量类型
using Vector3D = std::array<double, 3>;

// 向量点积
double dotProduct(const Vector3D &v1, const Vector3D &v2)
{
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

// 向量叉积
Vector3D crossProduct(const Vector3D &v1, const Vector3D &v2)
{
    return {
        v1[1] * v2[2] - v1[2] * v2[1], // i: bf - ce
        v1[2] * v2[0] - v1[0] * v2[2], // j: cd - af
        v1[0] * v2[1] - v1[1] * v2[0]  // k: ae - bd
    };
}

// 向量模长
double vectorNorm(const Vector3D &v)
{
    return std::sqrt(dotProduct(v, v));
}

// 向量归一化
Vector3D normalize(const Vector3D &v)
{
    double norm = vectorNorm(v);
    if (norm < 1e-10)
        return {0.0, 0.0, 0.0};
    return {v[0] / norm, v[1] / norm, v[2] / norm};
}

// 局部坐标系类
class LocalCoordinateSystem
{
private:
    Vector3D P;          // 原点
    Vector3D u1, u2, u3; // 正交基

public:
    LocalCoordinateSystem(const Vector3D &n1, const Vector3D &n2, const Vector3D &origin)
        : P(origin)
    {
        u1 = normalize(n1);
        u2 = normalize(n2);
        u3 = normalize(crossProduct(n1, n2));
    }

    // 全局坐标转局部坐标
    Vector3D globalToLocal(const Vector3D &q_global) const
    {
        Vector3D r = {q_global[0] - P[0], q_global[1] - P[1], q_global[2] - P[2]};
        return {dotProduct(r, u1), dotProduct(r, u2), dotProduct(r, u3)};
    }

    // 局部坐标转全局坐标
    Vector3D localToGlobal(const Vector3D &q_local) const
    {
        return {P[0] + q_local[0] * u1[0] + q_local[1] * u2[0] + q_local[2] * u3[0],
                P[1] + q_local[0] * u1[1] + q_local[1] * u2[1] + q_local[2] * u3[1],
                P[2] + q_local[0] * u1[2] + q_local[1] * u2[2] + q_local[2] * u3[2]};
    }
};

// -------------------------------------
int main(int argc, const char *argv[])
{
    // 示例1：标准坐标系平移
    Vector3D              n1 = {1.0, 0.0, 0.0};
    Vector3D              n2 = {0.0, 1.0, 0.0};
    Vector3D              P  = {2.0, 3.0, 4.0};
    LocalCoordinateSystem cs(n1, n2, P);

    // 测试点变换
    Vector3D global_point = {3.0, 4.0, 5.0};
    Vector3D local_point  = cs.globalToLocal(global_point);
    std::cout << "Global (3,4,5) -> Local (" << local_point[0] << "," << local_point[1] << "," << local_point[2]
              << ")\n";

    // 示例2：旋转45度的坐标系
    // double                theta  = M_PI / 4;
    double                theta  = std::numbers::pi / 4;
    Vector3D              n1_rot = {std::cos(theta), std::sin(theta), 0.0};
    Vector3D              n2_rot = {-std::sin(theta), std::cos(theta), 0.0};
    LocalCoordinateSystem rotated_cs(n1_rot, n2_rot, {0, 0, 0});

    Vector3D point         = {1.0, 0.0, 0.0};
    Vector3D rotated_point = rotated_cs.globalToLocal(point);
    std::cout << "Rotated system: Global (1,0,0) -> Local (" << rotated_point[0] << "," << rotated_point[1] << ","
              << rotated_point[2] << ")";

    return 0;
}
