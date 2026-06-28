#include "Light.h"

namespace Ithaca {

// 对于点 P, 计算其 L(p)
Color DirectionalLight::GetRadiance(const Vector3f &p, Vector3f &sourcePos) const
{
    // 假设光源在无限远处, 沿着光线方向
    sourcePos = p - Direction_ * 10000000.0f;

    return Radiance_;
}

// 对于点 P, 计算其 L(p)
Color PointLight::GetRadiance(const Vector3f &p, Vector3f &sourcePos) const
{
    sourcePos   = Position_;
    float R     = glm::length(p - Position_);
    float denom = Attenuations_.z + Attenuations_.y * R + Attenuations_.x * R * R;

    const float EPS   = 1e-6f;                // 与光源尺寸相关的极小值
    denom             = std::max(denom, EPS); // 保证至少为 EPS
    float attenuation = 1.0f / denom;

    // float attenuation = (denom < 1e-6f) ? 1.0f : (1.0f / denom);

    return Intensity_ * attenuation;
}

// 对于点 P, 计算其 L(p)
Color SpotLight::GetRadiance(const Vector3f &p, Vector3f &sourcePos) const
{
    sourcePos = Position_;
    // 距离衰减 K1
    float       R     = glm::length(p - Position_);
    float       denom = Attenuations_.z + Attenuations_.y * R + Attenuations_.x * R * R;
    const float EPS   = 1e-6f;                // 与光源尺寸相关的极小值
    denom             = std::max(denom, EPS); // 保证至少为 EPS
    float K1          = 1.0f / denom;

    // 角度衰减
    Vector3f L        = glm::normalize(p - Position_);
    float    cosTheta = glm::dot(L, Direction_);
    float    K2       = (cosTheta - CosOuterAngle_) / (CosInnerAngle_ - CosOuterAngle_);

    return Intensity_ * K1 * glm::clamp(K2, 0.0f, 1.0f);
}
} // namespace Ithaca