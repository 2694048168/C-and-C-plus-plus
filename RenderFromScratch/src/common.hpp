/**
 * @file common.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Common utility functions and type definitions
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/string_cast.hpp>

namespace Ithaca {

using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector4f = glm::vec4;

using Vector2i = glm::ivec2;
using Vector3i = glm::ivec3;
using Vector4i = glm::ivec4;

using Matrix3x3 = glm::mat3;
using Matrix4x4 = glm::mat4;

const float PI = glm::pi<float>();

using Color = glm::vec3;

inline Matrix4x4 MakeTranslateMatrix(const Vector3f &translation)
{
    // return glm::translate(Matrix4x4(1.0f), translation);
    // column-major order
    // clang-format off
    return Matrix4x4{
        1.0f,          0.0f,          0.0f,          0.0f, 
        0.0f,          1.0f,          0.0f,          0.0f,
        0.0f,          0.0f,          1.0f,          0.0f, 
        translation.x, translation.y, translation.z, 1.0f};
    // clang-format on
}

inline Matrix4x4 MakeRotateMatrix(const Vector3f &ruler)
{
    // return glm::rotate(Matrix4x4(1.0f), ruler.x, Vector3f(1.0f, 0.0f, 0.0f))
    //      * glm::rotate(Matrix4x4(1.0f), ruler.y, Vector3f(0.0f, 1.0f, 0.0f))
    //      * glm::rotate(Matrix4x4(1.0f), ruler.z, Vector3f(0.0f, 0.0f, 1.0f));

    // rotate X, then Y, then Z
    float cx = cosf(ruler.x);
    float sx = sinf(ruler.x);
    float cy = cosf(ruler.y);
    float sy = sinf(ruler.y);
    float cz = cosf(ruler.z);
    float sz = sinf(ruler.z);

    // clang-format off
    Matrix4x4 rx(
        1.0f, 0.0f, 0.0f, 0.0f, 
        0.0f, cx,   -sx,  0.0f, 
        0.0f, sx,   cx,   0.0f, 
        0.0f, 0.0f, 0.0f, 1.0f);

    Matrix4x4 ry(
        cy,   0.0f, sy,   0.0f, 
        0.0f, 1.0f, 0.0f, 0.0f, 
        -sy,  0.0f, cy,   0.0f, 
        0.0f, 0.0f, 0.0f, 1.0f);

    Matrix4x4 rz(
        cz,   -sz,  0.0f, 0.0f, 
        sz,   cz,   0.0f, 0.0f, 
        0.0f, 0.0f, 1.0f, 0.0f, 
        0.0f, 0.0f, 0.0f, 1.0f);
    // clang-format on

    return rz * ry * rx;
}

inline Matrix4x4 MakeScaleMatrix(const float &scale)
{
    // clang-format off
    return Matrix4x4{
        scale, 0.0f,  0.0f,  0.0f, 
        0.0f,  scale, 0.0f,  0.0f, 
        0.0f,  0.0f,  scale, 0.0f, 
        0.0f,  0.0f,  0.0f,  1.0f};
    // clang-format on
}

} // namespace Ithaca
