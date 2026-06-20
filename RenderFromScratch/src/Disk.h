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

#include "Primitive.h"

namespace Ithaca {

class Disk : public Primitive
{
private:
    float Radius_;

public:
    // Disk(SceneObject *pSceneObject, const Vector3f &center, const Vector3f &euler, float radius);
    Disk(SceneObject *pSceneObject, float radius);

    virtual bool Intersect(Ray ray, Intersection &isect) const override;
};

} // namespace Ithaca
