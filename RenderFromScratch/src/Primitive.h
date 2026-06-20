/**
 * @file Primitive.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Ray.h"

namespace Ithaca {

class SceneObject;

class Primitive
{
protected:
    SceneObject *pSceneObject_ = nullptr;

public:
    virtual bool Intersect(Ray ray, Intersection &isect) const = 0;

public:
    Primitive(SceneObject *pSceneObject)
        : pSceneObject_(pSceneObject)
    {
    }

    virtual ~Primitive() {}
};

} // namespace Ithaca
