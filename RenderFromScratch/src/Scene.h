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
#include "SceneObject.h"

namespace Ithaca {

class Scene
{
private:
    Camera                     camera_;
    std::vector<SceneObject *> sceneObjectVec_;

public:
    static Scene *LoadSceneFromXML(const char *filepath, int W, int H);

    void          SetCamera(const Camera &cam);
    const Camera &GetCamera() const;

    SceneObject *CreateSceneObject(const Vector3f &postion, const Vector3f &euler, float scale);

    SceneObject *Intersect(Ray ray, Intersection &isect) const;

public:
    Scene();
    virtual ~Scene();
};

} // namespace Ithaca