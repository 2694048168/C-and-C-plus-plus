#include "Scene.h"

#include "Sphere.h"
#include "Triangle.h"
#include "tinyxml2.h"

#include <cstdio>
#include <cstdlib>

namespace Ithaca {

Scene::Scene()
{
    sceneObjectVec_.clear();
}

Scene::~Scene()
{
    for (auto &sceneObj : sceneObjectVec_)
    {
        if (sceneObj)
        {
            delete sceneObj;
            sceneObj = nullptr;
        }
    }
    sceneObjectVec_.clear();
}

void Scene::SetCamera(const Camera &cam)
{
    camera_ = cam;
}

const Camera &Scene::GetCamera() const
{
    return camera_;
}

SceneObject *Scene::CreateSceneObject(const Vector3f &postion, const Vector3f &euler, float scale)
{
    SceneObject *pSceneObj = new SceneObject(postion, euler, scale);
    sceneObjectVec_.emplace_back(pSceneObj);
    return pSceneObj;
}

SceneObject *Scene::Intersect(Ray ray, Intersection &isect) const
{
    SceneObject *pHitObject = nullptr;
    for (const auto &sceneObj : sceneObjectVec_)
    {
        if (sceneObj->Intersect(ray, isect))
        {
            ray.max_t  = isect.t;
            pHitObject = sceneObj;
        }
    }
    return pHitObject;
}

// 解析 "x, y, z" 格式的字符串为 Vector3f
static Vector3f ParseVector3f(const char *text)
{
    if (!text)
        return Vector3f(0.0f);

    float x, y, z;
    if (sscanf_s(text, "%f, %f, %f", &x, &y, &z) == 3)
        return Vector3f(x, y, z);

    return Vector3f(0.0f);
}

Scene *Scene::LoadSceneFromXML(const char *filepath, int W, int H)
{
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError    error = doc.LoadFile(filepath);
    if (error != tinyxml2::XML_SUCCESS)
    {
        return nullptr;
    }

    Scene *pScene = new Scene();

    // -------------------- 解析 Camera --------------------
    tinyxml2::XMLElement *pRoot   = doc.RootElement();
    tinyxml2::XMLElement *pCamera = pRoot->FirstChildElement("Camera");
    if (pCamera)
    {
        Vector3f pos    = ParseVector3f(pCamera->FirstChildElement("Postion")->GetText());
        Vector3f target = ParseVector3f(pCamera->FirstChildElement("Target")->GetText());
        Vector3f up     = ParseVector3f(pCamera->FirstChildElement("Up")->GetText());
        float    nearZ  = pCamera->FirstChildElement("NearZ")->FloatText(0.1f);
        float    farZ   = pCamera->FirstChildElement("FarZ")->FloatText(1000.0f);
        float    fov    = pCamera->FirstChildElement("Fov")->FloatText(60.0f);

        Camera cam;
        cam.Initialize(pos, target, up, glm::radians(fov), nearZ, farZ, W, H);
        pScene->SetCamera(cam);
    }

    // -------------------- 解析 SceneObjectVec --------------------
    tinyxml2::XMLElement *pSceneObjectVec = pRoot->FirstChildElement("SceneObjectVec");
    if (pSceneObjectVec)
    {
        for (tinyxml2::XMLElement *pObj = pSceneObjectVec->FirstChildElement("SceneObject"); pObj != nullptr;
             pObj                       = pObj->NextSiblingElement("SceneObject"))
        {
            // --- 解析 Transform ---
            tinyxml2::XMLElement *pTransform = pObj->FirstChildElement("Transform");
            tinyxml2::XMLElement *pPosition  = pTransform->FirstChildElement("Position");
            tinyxml2::XMLElement *pRotation  = pTransform->FirstChildElement("Rotation");
            tinyxml2::XMLElement *pScale     = pTransform->FirstChildElement("Scale");

            Vector3f position = ParseVector3f(pPosition->GetText());
            Vector3f rotation = ParseVector3f(pRotation->GetText());
            float    scale    = pScale->FloatText(1.0f);

            SceneObject *pSceneObj = pScene->CreateSceneObject(position, rotation, scale);

            // --- 解析 PrimitiveVec ---
            tinyxml2::XMLElement *pPrimitiveVec = pObj->FirstChildElement("PrimitiveVec");
            if (pPrimitiveVec)
            {
                // 遍历 Triangle
                for (tinyxml2::XMLElement *pTri = pPrimitiveVec->FirstChildElement("Triangle"); pTri != nullptr;
                     pTri                       = pTri->NextSiblingElement("Triangle"))
                {
                    tinyxml2::XMLElement *pV0 = pTri->FirstChildElement("Vertex");
                    tinyxml2::XMLElement *pV1 = pV0->NextSiblingElement("Vertex");
                    tinyxml2::XMLElement *pV2 = pV1->NextSiblingElement("Vertex");

                    Vector3f v0 = ParseVector3f(pV0->GetText());
                    Vector3f v1 = ParseVector3f(pV1->GetText());
                    Vector3f v2 = ParseVector3f(pV2->GetText());

                    pSceneObj->CreatePrimitive<Triangle>(v0, v1, v2);
                }

                // 遍历 Sphere
                for (tinyxml2::XMLElement *pSphere = pPrimitiveVec->FirstChildElement("Sphere"); pSphere != nullptr;
                     pSphere                       = pSphere->NextSiblingElement("Sphere"))
                {
                    float radius = pSphere->FirstChildElement("Radius")->FloatText(1.0f);
                    pSceneObj->CreatePrimitive<Sphere>(radius);
                }
            }
        }
    }

    return pScene;
}

} // namespace Ithaca