/**
 * @file Graphics3D.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-08-29
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <cmath>

/**
 * @brief 3D坐标系系统
 * 在3D图形中, 通常使用右手坐标系:
 * X轴: 水平方向（右为正）
 * Y轴: 垂直方向（上为正）
 * Z轴: 深度方向（前为正）
 * 
 */

// 3D点结构
struct Vector3
{
    float x, y, z;

    Vector3(float x = 0, float y = 0, float z = 0)
        : x(x)
        , y(y)
        , z(z)
    {
    }

    // 向量运算
    Vector3 operator+(const Vector3 &other) const
    {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }

    Vector3 operator-(const Vector3 &other) const
    {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }

    // 点积
    float dot(const Vector3 &other) const
    {
        return x * other.x + y * other.y + z * other.z;
    }

    // 叉积
    Vector3 cross(const Vector3 &other) const
    {
        return Vector3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
    }

    // 归一化
    Vector3 normalize() const
    {
        float length = std::sqrt(x * x + y * y + z * z);
        return Vector3(x / length, y / length, z / length);
    }
};

/**
 * @brief 矩阵变换
 * 3D图形中的基本变换：平移、旋转、缩放
 * 
 */
// 4x4矩阵类
class Matrix4x4
{
public:
    float m[4][4];

    Matrix4x4()
    {
        // 初始化为单位矩阵
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) m[i][j] = (i == j) ? 1.0f : 0.0f;
    }

    // 平移矩阵
    static Matrix4x4 translation(float tx, float ty, float tz)
    {
        Matrix4x4 mat;
        mat.m[0][3] = tx;
        mat.m[1][3] = ty;
        mat.m[2][3] = tz;
        return mat;
    }

    // 缩放矩阵
    static Matrix4x4 scaling(float sx, float sy, float sz)
    {
        Matrix4x4 mat;
        mat.m[0][0] = sx;
        mat.m[1][1] = sy;
        mat.m[2][2] = sz;
        return mat;
    }

    // 绕X轴旋转
    static Matrix4x4 rotationX(float angle)
    {
        float cosA = cos(angle);
        float sinA = sin(angle);

        Matrix4x4 mat;
        mat.m[1][1] = cosA;
        mat.m[1][2] = -sinA;
        mat.m[2][1] = sinA;
        mat.m[2][2] = cosA;
        return mat;
    }

    // 矩阵乘法
    Matrix4x4 operator*(const Matrix4x4 &other) const
    {
        Matrix4x4 result;
        for (int i = 0; i < 4; ++i)
        {
            for (int j = 0; j < 4; ++j)
            {
                result.m[i][j] = 0;
                for (int k = 0; k < 4; ++k)
                {
                    result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
        }
        return result;
    }
};

// --------------------------------------
int main(int argc, const char *argv[])
{
    return 0;
}
