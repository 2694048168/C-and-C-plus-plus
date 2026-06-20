/**
 * @file Sphere.h
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

class Sphere : public Primitive
{
private:
    float Radius_;

public:
    Sphere(SceneObject *pSceneObject, float R);

    virtual bool Intersect(Ray ray, Intersection &isect) const override;
};

} // namespace Ithaca
