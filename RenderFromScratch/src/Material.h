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
};

class LambertMaterial : public Material
{
public:
    LambertMaterial(const Color &albedo);

    Color BRDF(const Vector3f &wo, const Vector3f &wi) const override;

private:
    Color mAlbedo;
};

} // namespace Ithaca