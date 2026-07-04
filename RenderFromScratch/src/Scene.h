/**
 * @file Scene.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-20
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Camera.h"
#include "Light.h"
#include "SceneObject.h"

#include <map>

namespace Ithaca {

class Scene
{
private:
    Camera                     camera_;
    std::vector<SceneObject *> sceneObjectVec_;
    std::vector<Light *>       lightVec_;

    std::map<std::string, Material *> mMaterialMap;

public:
    static Scene *LoadSceneFromXML(const char *filepath, int W, int H);

    void          SetCamera(const Camera &cam);
    const Camera &GetCamera() const;

    SceneObject *CreateSceneObject(const Vector3f &postion, const Vector3f &euler, float scale);

    SceneObject *Intersect(Ray ray, Intersection &isect) const;

    template<typename T, typename... Args>
    T *CreateLight(Args &&...args)
    {
        Light *light = new T(std::forward<Args>(args)...);
        lightVec_.emplace_back(light);
        return (T *)light;
    }

    const std::vector<Light *> GetLights() const
    {
        return lightVec_;
    }

    template<typename T, typename... Args>
    T *CreateMaterial(const std::string &nameID, Args &&...args)
    {
        T *material = new T(std::forward<Args>(args)...);
        mMaterialMap.insert({nameID, material});
        return material;
    }

    Material *GetMaterial(const std::string &name) const;

public:
    Scene();
    virtual ~Scene();
};

} // namespace Ithaca