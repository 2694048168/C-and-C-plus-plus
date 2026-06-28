/**
 * @file Renderer.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Camera.h"
#include "Disk.h"
#include "Primitive.h"
#include "Scene.h"
#include "SceneObject.h"
#include "Sphere.h"
#include "Triangle.h"
#include "common.hpp"

#include <MiniFB.h>

#include <atomic>
#include <string>
#include <string_view>
#include <vector>

namespace Ithaca {

class Renderer
{
public:
    Renderer() = default;
    explicit Renderer(const std::string_view &title, int width, int height, int SamplePerPixel, const char *filepath);
    virtual ~Renderer();

    void Run();

protected:
    Color RenderPixel(int x, int y);
    Color RenderSubPixel(float x, float y);
    Color GetIrradiance(const Ray &ray);

    void RunRenderThread();

private:
    std::string title_;
    int         viewportWidth_;
    int         viewportHeight_;
    // SPP algormthm
    int         SamplePerPixel_ = 100;

    struct mfb_window *pWindow_;
    uint32_t          *pBuffer_;
    std::atomic<int>   currentPixelIndex_;

    // Camera camera_;

    // Sphere   *pSphere   = nullptr;
    // Disk     *pDisk     = nullptr;
    // Triangle *pTriangle = nullptr;

    // std::vector<Primitive *> PrimitiveVec_;

    // SceneObject *pSceneObject_ = nullptr;

    Scene *pScene = nullptr;
};

} // namespace Ithaca
