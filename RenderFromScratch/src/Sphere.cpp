#include "Sphere.h"

namespace Ithaca {
Sphere::Sphere(const Vector3f &center, float R)
    : Radius_(R)
{
    ObjectToWorld_ = MakeWorldTransform(center, Vector3f(0.f, 0.f, 0.f), 1.0f);
    WorldToObject_ = glm::inverse(ObjectToWorld_);
}

bool Sphere::Intersect(Ray ray, Intersection &isect) const
{
    // ray transform into Sphere loc-coord. space
    Ray r = WorldToObject_ * ray;

    float A = glm::dot(r.d, r.d);
    float B = 2.0f * glm::dot(r.d, r.o);
    float C = glm::dot(r.o, r.o) - Radius_ * Radius_;

    float delta = B * B - 4.0f * A * C;
    if (delta < 0.0f)
        return false; // No Intersection

    float sqrtDelta = std::sqrtf(delta);
    float t1        = (-B - sqrtDelta) / (2.0f * A);
    float t2        = (-B + sqrtDelta) / (2.0f * A);
    // t1 < t2

    if (t2 < r.min_t)
        return false;
    if (t1 > r.max_t)
        return false;

    float t = t1;
    if (t < r.min_t)
    {
        t = t2;
        if (t > r.max_t)
            return false;
    }

    Vector3f p = r.o + t * r.d;
    Vector3f n = glm::normalize(p);

    isect.postion = Vector3f(ObjectToWorld_ * Vector4f(p, 1.0f));
    isect.normal  = glm::normalize(ObjectToWorld_ * Vector4f(n, 0.0f));
    isect.t       = t;

    return true;
}

} // namespace Ithaca