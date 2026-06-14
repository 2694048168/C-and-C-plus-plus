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

} // namespace Ithaca
