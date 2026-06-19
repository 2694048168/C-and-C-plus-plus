#include "Triangle.h"

namespace Ithaca {
Triangle::Triangle(const Vector3f &v0, const Vector3f &v1, const Vector3f &v2, const Matrix4x4 &worldMatrix)
{
    Vertices_[0] = Vector3f(worldMatrix * Vector4f(v0, 1.0f));
    Vertices_[1] = Vector3f(worldMatrix * Vector4f(v1, 1.0f));
    Vertices_[2] = Vector3f(worldMatrix * Vector4f(v2, 1.0f));

    Vector3f edge1 = Vertices_[1] - Vertices_[0];
    Vector3f edge2 = Vertices_[2] - Vertices_[0];
    Normal_        = glm::normalize(glm::cross(edge1, edge2));
}

bool Triangle::Intersect(Ray ray, Intersection &isect) const
{
    Vector3f p0 = Vertices_[0];
    Vector3f p1 = Vertices_[1];
    Vector3f p2 = Vertices_[2];

    Vector3f e1 = p1 - p0;
    Vector3f e2 = p2 - p0;
    Vector3f s  = ray.o - p0;

    Vector3f s1 = glm::cross(ray.d, e2);
    Vector3f s2 = glm::cross(s, e1);

    float det = glm::dot(s1, e1);
    if (std::fabs(det) < 1e-6f)
        return false;
    float invDet = 1.0f / det;

    float b1 = glm::dot(s1, s) * invDet;
    float b2 = glm::dot(s2, ray.d) * invDet;
    float t  = glm::dot(s2, e2) * invDet;

    if (t < ray.min_t || t > ray.max_t)
        return false;

    float b0 = 1.0f - b1 - b2;
    if (b0 < 0.0f || b1 < 0.0f || b2 < 0.0f)
        return false;

    isect.postion = ray.o + t * ray.d;
    isect.normal  = Normal_;
    isect.t       = t;

    return true;
}

} // namespace Ithaca