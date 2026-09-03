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

// 反射
Color DielectricSpecularMaterial::BRDF(const Vector3f &wo, const Vector3f &wi) const
{
    float cos_i = std::fabs(wi.z);
    float eta_i, eta_t;
    if (wi.z > 0)
    {
        eta_i = 1.0f;
        eta_t = mEta;
    }
    else
    {
        eta_i = mEta;
        eta_t = 1.0f;
    }

    Vector3f wt;
    float    Fr = 1.0f;
    if (ComputeRefractVector(wi, eta_i, eta_t, wt))
    {
        float cos_t = std::fabs(wt.z);
        Fr          = Fresnel(eta_i, eta_t, cos_i, cos_t);
    }

    return Color(Fr) / std::max(fabs(wi.z), 1e-4f);
}

// 折射
Color DielectricSpecularMaterial::BTDF(const Vector3f &wt, const Vector3f &wi) const
{
    float eta_i, eta_t;
    if (wi.z > 0)
    {
        eta_i = 1.0f;
        eta_t = mEta;
    }
    else
    {
        eta_i = mEta;
        eta_t = 1.0f;
    }

    float cos_i = std::fabs(wi.z);
    float cos_t = std::fabs(wt.z);

    float Fr = Fresnel(eta_i, eta_t, cos_i, cos_t);

    return mTransmissionColor * (1.0f - Fr) * eta_t * eta_t / (eta_i * eta_i * std::max(1e-4f, cos_i));
}

float DielectricSpecularMaterial::Fresnel(float eta_i, float eta_t, float cos_i, float cos_t)
{
    float r1 = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t);
    float r2 = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t);
    return 0.5f * (r1 * r1 + r2 * r2);
}

bool DielectricSpecularMaterial::SampleWt(const Vector3f &wi, Vector3f &wt) const
{
    float cos_i = std::fabs(wi.z);
    float eta_i, eta_t;
    if (wi.z > 0)
    {
        eta_i = 1.0f;
        eta_t = mEta;
    }
    else
    {
        eta_i = mEta;
        eta_t = 1.0f;
    }

    return ComputeRefractVector(wi, eta_i, eta_t, wt);
}

} // namespace Ithaca