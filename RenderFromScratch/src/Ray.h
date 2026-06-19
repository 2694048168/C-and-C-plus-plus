/**
 * @file Ray.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "common.hpp"

#include <numeric>

namespace Ithaca {

struct Ray
{
    Vector3f o;
    Vector3f d;

    float min_t = 0.0f;
    float max_t = std::numeric_limits<float>::max();
};

inline Ray operator*(const Matrix4x4 &m, const Ray &r)
{
    Ray result;

    result.o     = Vector3f(m * Vector4f(r.o, 1.0f));
    result.d     = Vector3f(m * Vector4f(r.d, 0.0f));
    result.min_t = r.min_t;
    result.max_t = r.max_t;

    return result;
}

struct Intersection
{
    Vector3f postion;
    Vector3f normal;
    float    t;
};

} // namespace Ithaca
