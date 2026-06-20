#include "Sphere.h"

#include "SceneObject.h"

namespace Ithaca {
Sphere::Sphere(SceneObject *pSceneObject, float R)
    : Primitive(pSceneObject)
    , Radius_(R)
{
}

bool Sphere::Intersect(Ray ray, Intersection &isect) const
{
    // ray transform into Sphere loc-coord. space
    Ray r = pSceneObject_->GetWorldToObject() * ray;

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

    isect.postion = Vector3f(pSceneObject_->GetObjectToWorld() * Vector4f(p, 1.0f));
    isect.normal  = glm::normalize(pSceneObject_->GetObjectToWorld() * Vector4f(n, 0.0f));
    isect.t       = t;

    return true;
}

} // namespace Ithaca