/**
 * @file GLM_Example.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief GLM tutorial example
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "logger.hpp"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
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

int main(int argc, const char **argv)
{
    {
        Ithaca::Vector3f vec1(1.0f, 2.0f, 3.0f);
        Ithaca::Vector3f vec2(4.0f, 5.0f, 6.0f);

        Ithaca::Vector3f sum  = vec1 + vec2;
        Ithaca::Vector3f diff = vec1 - vec2;
        Ithaca::Logger::log("\nVector Sum: " + glm::to_string(sum));
        Ithaca::Logger::log("\nVector Diff: " + glm::to_string(diff));

        float            dotProduct   = glm::dot(vec1, vec2);
        Ithaca::Vector3f crossProduct = glm::cross(vec1, vec2);
        Ithaca::Logger::log("\nVector Dot Product: " + std::to_string(dotProduct));
        Ithaca::Logger::log("\nVector Cross Product: " + glm::to_string(crossProduct));

        float            lenth         = glm::length(vec1);
        Ithaca::Vector3f normalizedVec = glm::normalize(vec1);
        Ithaca::Logger::log("\nVector Length: " + std::to_string(lenth));
        Ithaca::Logger::log("\nVector Normalize: " + glm::to_string(normalizedVec));
    }
    {
        Ithaca::Matrix4x4 mat1 = glm::mat4(1.0f);
        Ithaca::Logger::log("\nIdentity Matrix: " + glm::to_string(mat1));

        // column-major order
        /*
        [1, 2, 3, 4
         5, 6, 7, 8,
         9, 10, 11, 12,
         13, 14, 15, 16]
        */
        Ithaca::Matrix4x4 mat2
            = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

        Ithaca::Matrix4x4 matTranspose = glm::transpose(mat2);
        Ithaca::Logger::log("\nTranspose Matrix: " + glm::to_string(matTranspose));

        Ithaca::Matrix4x4 matInverse = glm::inverse(mat2);
        Ithaca::Logger::log("\nInverse Matrix: " + glm::to_string(matInverse));
    }

    {
        Ithaca::Vector3f pos1(1.0f, 2.0f, 3.0f);
        Ithaca::Vector3f transaction1(4.0f, 5.0f, 6.0f);
        Ithaca::Vector3f newPos = pos1 + transaction1;
        Ithaca::Logger::log("\nNew Position: " + glm::to_string(newPos));

        // translation matrix
        Ithaca::Matrix4x4 translateMat = Ithaca::MakeTranslateMatrix(transaction1);
        Ithaca::Vector4f  posTranspose = translateMat * Ithaca::Vector4f(pos1, 1.0f);
        Ithaca::Logger::log("\nTranspose Position: " + glm::to_string(posTranspose));
    }

    {
        // Ithaca::Matrix4x4 R = Ithaca::MakeRotateMatrix(
        //     Ithaca::Vector3f(30.0f / 180.0f * Ithaca::PI, 45.0f / 180.0f * Ithaca::PI, 60.0f / 180.0f * Ithaca::PI));

        Ithaca::Matrix4x4 R
            = Ithaca::MakeRotateMatrix(Ithaca::Vector3f(glm::radians(30.0f), glm::radians(45.0f), glm::radians(60.0f)));

        // rotate vertor
        // Ithaca::Vector4f rotatedVector = R * Ithaca::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
        Ithaca::Vector3f rotatedVector = R * Ithaca::Vector4f(1.0f, 0.0f, 0.0f, 0.0f);
        // rotate point
        // Ithaca::Vector4f rotatedPoint = R * Ithaca::Vector4f(1.0f, 0.0f, 0.0f, 1.0f);
        Ithaca::Vector3f rotatedPoint = R * Ithaca::Vector4f(1.0f, 0.0f, 0.0f, 1.0f);

        Ithaca::Logger::log("\nRotated Vector: " + glm::to_string(rotatedVector));
        Ithaca::Logger::log("\nRotated Point: " + glm::to_string(rotatedPoint));
    }

    {
        Ithaca::Matrix4x4 S            = Ithaca::MakeScaleMatrix(2.0f);
        Ithaca::Vector3f  scaledVector = S * Ithaca::Vector4f(1.0f, 2.0f, 3.0f, 0.0f);
        Ithaca::Vector3f  scaledPoint  = S * Ithaca::Vector4f(1.0f, 2.0f, 3.0f, 1.0f);

        Ithaca::Logger::log("\nScale Vector: " + glm::to_string(scaledVector));
        Ithaca::Logger::log("\nScale Point: " + glm::to_string(scaledPoint));
    }

    return 0;
}
