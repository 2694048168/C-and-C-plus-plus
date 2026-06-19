/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "Renderer.h"
#include "logger.hpp"
#include "version.h"

int main(int argc, const char **argv)
{
    std::string versionStr = Ithaca::Version::getVersionBuild();
    Ithaca::Logger::log("\nApplication Version: " + versionStr);

    std::string   titleStr = "Ithaca Render Engine " + versionStr;
    constexpr int width    = 1080; // 1920
    constexpr int height   = 720; // 1080

    Ithaca::Renderer renderEngine(titleStr, width, height);
    renderEngine.Run();

    return 0;
}
