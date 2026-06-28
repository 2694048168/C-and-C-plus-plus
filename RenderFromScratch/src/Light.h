/**
 * @file Light.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-28
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Ray.h"

namespace Ithaca {

class Light
{
public:
    // 对于点 P, 计算其 L(p)
    virtual Color GetRadiance(const Vector3f &p, Vector3f &sourcePos) const = 0;

public:
    Light()          = default;
    virtual ~Light() = default;
};

// 平行光 Directional Light 3D
class DirectionalLight : public Light
{
private:
    // 光线方向, 单位向量
    Vector3f Direction_;
    // 光照强度
    Color    Radiance_;

public:
    // 对于点 P, 计算其 L(p)
    Color GetRadiance(const Vector3f &p, Vector3f &sourcePos) const override;

public:
    DirectionalLight(const Vector3f &direction, const Color &radiance)
        : Direction_(glm::normalize(direction))
        , Radiance_(radiance)
    {
    }

    ~DirectionalLight() = default;
};

// 点光源 Spot Light 3D
class PointLight : public Light
{
private:
    // 光源位置
    Vector3f Position_;
    // 光照强度
    Color    Intensity_;
    // 衰减系数, A-B-C
    Vector3f Attenuations_;

public:
    // 对于点 P, 计算其 L(p)
    Color GetRadiance(const Vector3f &p, Vector3f &sourcePos) const override;

public:
    PointLight(const Vector3f &position, const Color &intensity, const Vector3f &attenuataions)
        : Position_(position)
        , Intensity_(intensity)
        , Attenuations_(attenuataions)
    {
    }

    ~PointLight() = default;
};

// 聚光灯 Omni or Spot Light 3D
class SpotLight : public Light
{
private:
    // 光线方向, 单位向量
    Vector3f Direction_;
    // 光源位置
    Vector3f Position_;
    // 光照强度
    Color    Intensity_;
    // 衰减系数, A-B-C
    Vector3f Attenuations_;

    // 内锥角(的一半 alpha), 外锥角(的一半 beta)
    float CosInnerAngle_;
    float CosOuterAngle_;

public:
    // 对于点 P, 计算其 L(p)
    Color GetRadiance(const Vector3f &p, Vector3f &sourcePos) const override;

public:
    SpotLight(const Vector3f &position, const Vector3f &direction, const Color &intensity, float innerAngle,
              float outerAngle, const Vector3f &attenuataions)
        : Position_(position)
        , Direction_(direction)
        , Intensity_(intensity)
        , CosInnerAngle_(std::cos(innerAngle))
        , CosOuterAngle_(std::cos(outerAngle))
        , Attenuations_(attenuataions)
    {
    }

    ~SpotLight() = default;
};

} // namespace Ithaca
