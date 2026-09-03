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

    std::string   titleStr       = "Ithaca Render Engine " + versionStr;
    constexpr int width          = 1080; // 1920 & 1080
    constexpr int height         = 720;  // 1080 & 720
    constexpr int samplePerPixel = 100;  // SPP
    constexpr int maxDepth       = 10;
    constexpr int minDepth       = 3;
    // const char   *filepath       = "data/Scene01.xml";
    // const char   *filepath       = "data/Scene02CornellBox.xml";
    // const char   *filepath       = "data/Scene03CornellBox.xml";
    // const char   *filepath = "data/Scene04CornellBox.xml";
    // const char   *filepath = "data/Scene05CornellBox.xml";
    // const char   *filepath = "data/Scene06CornellBox.xml";
    const char   *filepath = "data/Scene08.xml";

    Ithaca::Renderer renderEngine(titleStr, width, height, minDepth, maxDepth, samplePerPixel, filepath);
    renderEngine.Run();

    return 0;
}
