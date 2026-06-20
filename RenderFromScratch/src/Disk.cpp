#include "Disk.h"

#include "SceneObject.h"

namespace Ithaca {
Disk::Disk(SceneObject *pSceneObject, float radius)
    : Primitive(pSceneObject)
    , Radius_(radius)
{
}

bool Disk::Intersect(Ray ray, Intersection &isect) const
{
    // ray transform into Sphere loc-coord. space
    Ray r = pSceneObject_->GetWorldToObject() * ray;

    if (std::fabs(r.d.z) < 1e-7f)
        return false;

    float t = -r.o.z / r.d.z;
    if (t < r.min_t || t > r.max_t)
        return false;

    Vector3f p = r.o + t * r.d;
    // postion inside Disk(Circle)
    if (glm::dot(p, p) > Radius_ * Radius_)
        return false;

    isect.postion = Vector3f(pSceneObject_->GetObjectToWorld() * Vector4f(p, 1.0f));
    isect.normal  = glm::normalize(pSceneObject_->GetObjectToWorld() * Vector4f(0.f, 0.f, 1.f, 0.0f));
    isect.t       = t;

    return true;
}

} // namespace Ithaca