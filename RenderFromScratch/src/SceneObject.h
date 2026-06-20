/**
 * @file SceneObject.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Primitive.h"

#include <vector>

namespace Ithaca {

class SceneObject
{
private:
    Matrix4x4 ObjectToWorld_;
    Matrix4x4 WorldToObject_;

    std::vector<Primitive *> PrimitiveVec_;

public:
    Matrix4x4 GetObjectToWorld() const;
    Matrix4x4 GetWorldToObject() const;

    // void AddPrimitive(Primitive *primitive);
    // Modern C++ RILL Tech.
    template<typename T, typename... Args>
    T *CreatePrimitive(Args &&...args)
    {
        T *primitive = new T(this, std::forward<Args>(args)...);
        PrimitiveVec_.emplace_back(primitive);
        return primitive;
    }

    bool Intersect(Ray ray, Intersection &isect) const;

public:
    SceneObject(const Vector3f &postion, const Vector3f &euler, float scale);
    virtual ~SceneObject();
};

} // namespace Ithaca
