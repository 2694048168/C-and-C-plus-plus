/**
 * @file Material.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-07-04
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "common.hpp"

namespace Ithaca {

class Material
{
public:
    virtual Color BRDF(const Vector3f &wo, const Vector3f &wi) const = 0;

    virtual Color BTDF(const Vector3f &wt, const Vector3f &wi) const
    {
        return Color(0);
    }

    virtual bool SampleWt(const Vector3f &wi, Vector3f &wt) const
    {
        return false;
    }

    virtual bool IsSpecular() const
    {
        return false;
    }
};

class LambertMaterial : public Material
{
public:
    LambertMaterial(const Color &albedo);

    Color BRDF(const Vector3f &wo, const Vector3f &wi) const override;

private:
    Color mAlbedo;
};

class ConductorSpecularMaterial : public Material
{
public:
    ConductorSpecularMaterial(const Color &eta, const Color &absorptionCoef, const Color &refectionColor)
        : mEta(eta)
        , mAbsorptionCoef(absorptionCoef)
        , mRefectionColor(refectionColor)
    {
    }

    bool IsSpecular() const override
    {
        return true;
    }

    Color BRDF(const Vector3f &wo, const Vector3f &wi) const override;

private:
    Color mEta;
    Color mAbsorptionCoef;
    Color mRefectionColor;
};

// 绝缘体 镜面材质
class DielectricSpecularMaterial : public Material
{
public:
    DielectricSpecularMaterial(const float &eta, const Color &transmissionColor)
        : mEta(eta)
        , mTransmissionColor(transmissionColor)
    {
    }

    bool IsSpecular() const override
    {
        return true;
    }

    // 反射
    Color BRDF(const Vector3f &wo, const Vector3f &wi) const override;
    // 折射
    Color BTDF(const Vector3f &wt, const Vector3f &wi) const override;

    bool SampleWt(const Vector3f &wi, Vector3f &wt) const override;

private:
    static float Fresnel(float eta_i, float eta_t, float cos_i, float cos_t);

private:
    float mEta;
    Color mTransmissionColor;
};

} // namespace Ithaca