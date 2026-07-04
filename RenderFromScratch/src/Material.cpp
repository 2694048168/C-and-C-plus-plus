#include "Material.h"

namespace Ithaca {

LambertMaterial::LambertMaterial(const Color &albedo)
    : mAlbedo(albedo)
{
}

Color LambertMaterial::BRDF(const Vector3f &wo, const Vector3f &wi) const
{
    return mAlbedo * INV_PI;
}

} // namespace Ithaca