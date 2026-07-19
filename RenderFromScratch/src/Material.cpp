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

Color ConductorSpecularMaterial::BRDF(const Vector3f &wo, const Vector3f &wi) const
{
    float delta = 1e-4f;
    if (!(fabs(wo.x + wi.x) < delta && fabs(wo.y + wi.y) < delta && fabs(wo.z - wi.z) < delta))
    {
        return Color(0.0f, 0.0f, 0.0f);
    }

    float cosTheta = std::max(delta, wi.z);
    Color k        = mAbsorptionCoef;
    Color r1       = ((mEta * mEta + k * k) * cosTheta * cosTheta - 2.0f * mEta * cosTheta + 1.0f)
                   / ((mEta * mEta + k * k) * cosTheta * cosTheta + 2.0f * mEta * cosTheta + 1.0f);
    Color r2       = ((mEta * mEta + k * k - 2.0f * mEta * cosTheta + cosTheta * cosTheta)
                      / (mEta * mEta + k * k + 2.0f * mEta * cosTheta + cosTheta * cosTheta));
    Color Fr       = (r1 + r2) * 0.5f;

    return Fr * mRefectionColor / std::max(fabs(cosTheta), delta);
}

} // namespace Ithaca