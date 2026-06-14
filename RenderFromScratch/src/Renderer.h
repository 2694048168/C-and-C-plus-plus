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
#include "common.hpp"

#include <MiniFB.h>

#include <atomic>
#include <string>
#include <string_view>

namespace Ithaca {

class Renderer
{
public:
    Renderer() = default;
    explicit Renderer(const std::string_view &title, int width, int height);
    virtual ~Renderer() = default;

    void Run();

protected:
    Color RenderPixel(int x, int y);

    void RunRenderThread();

private:
    std::string title_;
    int         viewportWidth_;
    int         viewportHeight_;

    struct mfb_window *pWindow_;
    uint32_t          *pBuffer_;
    std::atomic<int>   currentPixelIndex_;

    Camera camera_;
};

} // namespace Ithaca
