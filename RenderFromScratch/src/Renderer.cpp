#include "Renderer.h"

#include "logger.hpp"

#include <chrono>
#include <thread>
#include <vector>

namespace Ithaca {
Renderer::Renderer(const std::string_view &title, int width, int height)
    : title_(title)
    , viewportWidth_(width)
    , viewportHeight_(height)
    , pWindow_(nullptr)
    , pBuffer_(nullptr)
    , currentPixelIndex_(0)
{
    pWindow_ = mfb_open_ex(title_.c_str(), viewportWidth_, viewportHeight_, MFB_WF_RESIZABLE);
    pBuffer_ = (uint32_t *)malloc(viewportWidth_ * viewportHeight_ * 4);
    // Initialize buffer
    memset(pBuffer_, 0, viewportWidth_ * viewportHeight_ * 4);

    // clang-format off
    camera_.Initialize(
        Vector3f(0.0f, 0.0f, 0.0f), // camera postion
        Vector3f(0.0f, 0.0f, 1.0f), // object postion
        Vector3f(0.0f, 1.0f, 0.0f),
        glm::radians(60.0f), // fov
        0.1f,
        1000.0f,
        viewportWidth_,
        viewportHeight_
    );
    // clang-format on

    pSphere   = new Sphere(Vector3f(0.f, 0.f, 5.0f), 1.0f);
    pDisk     = new Disk(Vector3f(0.f, 0.f, 3.0f), Vector3f(0.f, 0.f, 0.0f), 1.0f);
    pTriangle = new Triangle(Vector3f(-1.f, 0.f, 0.f), Vector3f(0.f, 1.f, 0.f), Vector3f(1.f, 0.f, 0.f),
                             MakeWorldTransform(Vector3f(0.f, 0.f, 5.f), Vector3f(0.f, 0.f, 0.f), 2.0f));
}

void Renderer::Run()
{
    if (pWindow_ == NULL)
    {
        Ithaca::Logger::log("Failed to create window");
        return;
    }

    // !Single thread execution
    // std::thread renderThread(&Renderer::RunRenderThread, this);
    // renderThread.detach();

    // *multi-thread execution
    // auto                     numThreads = std::thread::hardware_concurrency();
    auto                     numThreads = 128;
    std::vector<std::thread> renderThreadVec(numThreads);
    for (size_t i = 0; i < numThreads; ++i)
    {
        renderThreadVec[i] = std::thread(&Renderer::RunRenderThread, this);
        renderThreadVec[i].detach();
    }

    mfb_update_state state;
    do
    {
        state = mfb_update_ex(pWindow_, pBuffer_, viewportWidth_, viewportHeight_);

        if (state != MFB_STATE_OK)
            break;
    }
    while (mfb_wait_sync(pWindow_));

    free(pBuffer_);
    pBuffer_ = NULL;
    pWindow_ = NULL;
}

Color Renderer::RenderPixel(int x, int y)
{
    // return Color(1.0f, 0.3f, 0.6f);

    // *Simple gradient effect
    // Color color;
    // color.r = x / (float)viewportWidth_;
    // color.g = y / (float)viewportHeight_;
    // color.b = 0.1f;

    // simulate a long computation 1ms
    // std::this_thread::sleep_for(std::chrono::milliseconds(1));

#if 0
    // ?Ray tracing implementation
    Ray ray = camera_.GenerateRay(x, y);
    // Color color = ray.d * 0.5f + 0.5f;
    // return color;

    Intersection isect;
    // *Rendering Sphere
    // if (pSphere->Intersect(ray, isect))
    // *Rendering Disk(Cirle in 3D space)
    // if (pDisk->Intersect(ray, isect))
    // *Rendering Triangle
    if (pTriangle->Intersect(ray, isect))
    {
        // return Color(1.f, 0.f, 0.f);

        Color color = isect.normal * 0.5f + 0.5f;
        return color;
    }
    return Color(0.f, 0.f, 0.f);
#endif

    // * SSAA algorithm
    // constexpr int N           = 4; // sample point
    constexpr int N           = 100; // sample point
    Color         resultColor = Color(0.f, 0.0f, 0.0f);

    for (int i = 0; i < N; ++i)
    {
        // (x, y) ~ (x+1, y+1) range random sample point
        float px = x + glm::linearRand(0.0f, 1.0f);
        float py = y + glm::linearRand(0.0f, 1.0f);

        Ray   ray   = camera_.GenerateRay(px, py);
        Color color = Color(0.f, 0.0f, 0.0f);

        Intersection isect;
        // *Rendering Sphere
        if (pSphere->Intersect(ray, isect))
        // *Rendering Disk(Cirle in 3D space)
        // if (pDisk->Intersect(ray, isect))
        // *Rendering Triangle
        // if (pTriangle->Intersect(ray, isect))
        {
            color += isect.normal * 0.5f + 0.5f;
        }

        // !overflow
        resultColor += (color / static_cast<float>(N));
    }

    return resultColor;
}

void Renderer::RunRenderThread()
{
    // !single thread render pass
    // for (int y = 0; y < viewportHeight_; ++y)
    // {
    //     for (size_t x = 0; x < viewportWidth_; ++x)
    //     {
    //         Color color = RenderPixel(x, y);

    //         uint32_t r = glm::clamp((uint32_t)std::round(color.r * 255), 0u, 255u);
    //         uint32_t g = glm::clamp((uint32_t)std::round(color.g * 255), 0u, 255u);
    //         uint32_t b = glm::clamp((uint32_t)std::round(color.b * 255), 0u, 255u);
    //         uint32_t a = 255;

    //         pBuffer_[y * viewportWidth_ + x] = (a << 24) | (r << 16) | (g << 8) | b;
    //     }
    // }

    // *parallel thread render pass
    // read window next pixel
    while (true)
    {
        // atomic operator
        int pixelIndex = currentPixelIndex_.fetch_add(1);
        if (pixelIndex >= viewportWidth_ * viewportHeight_)
            break;

        int x = pixelIndex % viewportWidth_;
        int y = pixelIndex / viewportWidth_;

        Color    color = RenderPixel(x, y);
        uint32_t r     = glm::clamp((uint32_t)std::round(color.r * 255), 0u, 255u);
        uint32_t g     = glm::clamp((uint32_t)std::round(color.g * 255), 0u, 255u);
        uint32_t b     = glm::clamp((uint32_t)std::round(color.b * 255), 0u, 255u);
        uint32_t a     = 255;

        pBuffer_[y * viewportWidth_ + x] = (a << 24) | (r << 16) | (g << 8) | b;
    }

    // TODO: Tile-Based Rendering
}
} // namespace Ithaca
