/**
 * @file Camera.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Ray.h"

namespace Ithaca {

class Camera
{
public:
    Camera()  = default;
    ~Camera() = default;

    void Initialize(const Vector3f &pos, const Vector3f &target, const Vector3f &up, float fov, float zNear, float zFar,
                    int W, int H);

    Ray GenerateRay(int x, int y) const;
    Ray GenerateRay(float x, float y) const;

private:
    Vector3f  position_;
    Matrix4x4 combinedMatrix_;
    Matrix4x4 invCombinedMatrix_;
};

} // namespace Ithaca