#include "SceneObject.h"

namespace Ithaca {

SceneObject::SceneObject(const Vector3f &postion, const Vector3f &euler, float scale)
{
    ObjectToWorld_ = MakeWorldTransform(postion, euler, scale);
    WorldToObject_ = glm::inverse(ObjectToWorld_);

    PrimitiveVec_.clear();
}

SceneObject::~SceneObject()
{
    for (auto &primitive : PrimitiveVec_)
    {
        if (primitive)
        {
            delete primitive;
            primitive = nullptr;
        }
    }
    PrimitiveVec_.clear();
}

Matrix4x4 SceneObject::GetObjectToWorld() const
{
    return ObjectToWorld_;
}

Matrix4x4 SceneObject::GetWorldToObject() const
{
    return WorldToObject_;
}

bool SceneObject::Intersect(Ray ray, Intersection &isect) const
{
    bool hitFlag = false;
    for (const auto &primitive : PrimitiveVec_)
    {
        if (primitive->Intersect(ray, isect))
        {
            ray.max_t = isect.t;
            hitFlag   = true;
        }
    }
    return hitFlag;
}

// void SceneObject::AddPrimitive(Primitive *primitive)
// {
//     PrimitiveVec_.emplace_back(primitive);
// }

} // namespace Ithaca
