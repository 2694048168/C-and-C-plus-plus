/**
 * @file MiniFB_Example.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief A simple example using MiniFB for rendering
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "logger.hpp"
#include "version.h"

#include <MiniFB.h>

int main(int argc, const char **argv)
{
    std::string versionStr = Ithaca::Version::getVersionBuild();
    Ithaca::Logger::log("\nApplication Version: " + versionStr);

    std::string titleStr = "Ithaca Render Engine " + versionStr;

    struct mfb_window *window = mfb_open_ex(titleStr.c_str(), 800, 600, MFB_WF_RESIZABLE);
    if (window == NULL)
        return 0;

    uint32_t *buffer = (uint32_t *)malloc(800 * 600 * 4);

    // Initialize buffer
    memset(buffer, 0, 800 * 600 * 4);
    for (int y = 0; y < 600; ++y)
    {
        for (size_t x = 0; x < 800; ++x)
        {
            uint32_t r = (x * 255);
            uint32_t g = (y * 255);
            uint32_t b = (x * y) % 255;
            uint32_t a = 255;

            buffer[y * 800 + x] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    mfb_update_state state;
    do
    {
        state = mfb_update_ex(window, buffer, 800, 600);

        if (state != MFB_STATE_OK)
            break;
    }
    while (mfb_wait_sync(window));

    free(buffer);
    buffer = NULL;
    window = NULL;

    return 0;
}
