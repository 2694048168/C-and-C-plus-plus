#include "Camera.h"

namespace Ithaca {

void Camera::Initialize(const Vector3f &pos, const Vector3f &target, const Vector3f &up, float fov, float zNear,
                        float zFar, int W, int H)
{
    position_ = pos;

    // ?view matrix
    // Vector3f l = glm::normalize(target - position_);
    // Vector3f r = glm::normalize(glm::cross(up, l));
    // Vector3f u = glm::cross(l, r);

    // Matrix4x4 viewMatrix = glm::transpose(
    //     Matrix4x4(
    //         r.x, r.y, r.z, 0.0f,
    //         u.x, u.y, u.z, 0.0f,
    //         l.x, l.y, l.z, 0.0f,
    //         0.0f, 0.0f, 0.0f, 1.0f
    //     )
    // ) * MakeTranslateMatrix(-pos);
    Matrix4x4 viewMatrix = glm::lookAtLH(pos, target, up);

    // ?Projection Matrix
    Matrix4x4 projectionMatrix
        = glm::perspectiveFovLH_ZO(fov, static_cast<float>(W), static_cast<float>(H), zNear, zFar);

    // ?viewport Matrix
    // clang-format off
    Matrix4x4 viewportMatrix = Matrix4x4(
        W * 0.5f,  0.0f,     0.0f, 0.0f,
        0.0f,     -H * 0.5f, 0.0f, 0.0f,
        0.0f,     0.0f,      1.0f, 0.0f,
        W * 0.5f, H * 0.5f,  0.f,  1.0f
    );
    // clang-format on

    combinedMatrix_    = viewportMatrix * projectionMatrix * viewMatrix;
    invCombinedMatrix_ = glm::inverse(combinedMatrix_);
}

Ray Camera::GenerateRay(int x, int y) const
{
    Ray ray;
    ray.o = position_;

    Vector4f p(static_cast<float>(x), static_cast<float>(y), 0.0f, 1.0f);
    Vector4f wordPos = glm::normalize(invCombinedMatrix_ * p);
    wordPos /= wordPos.w;

    ray.d = glm::normalize(Vector3f(wordPos) - position_);

    return ray;
}

} // namespace Ithaca
