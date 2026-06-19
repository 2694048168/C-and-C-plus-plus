/**
 * @file Disk.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-19
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Ray.h"

namespace Ithaca {

class Disk
{
private:
    float     Radius_;
    Matrix4x4 ObjectToWorld_;
    Matrix4x4 WorldToObject_;

public:
    Disk(const Vector3f &center, const Vector3f &euler, float radius);

    bool Intersect(Ray ray, Intersection &isect) const;
};

} // namespace Ithaca
