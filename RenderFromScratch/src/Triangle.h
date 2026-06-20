/**
 * @file Triangle.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-19
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Primitive.h"

namespace Ithaca {

class Triangle : public Primitive
{
private:
    Vector3f Vertices_[3];
    Vector3f Normal_;

public:
    // Triangle(SceneObject *pSceneObject,const Vector3f &v0, const Vector3f &v1, const Vector3f &v2, const Matrix4x4 &worldMatrix);
    Triangle(SceneObject *pSceneObject, const Vector3f &v0, const Vector3f &v1, const Vector3f &v2);

    virtual bool Intersect(Ray ray, Intersection &isect) const override;
};

} // namespace Ithaca
