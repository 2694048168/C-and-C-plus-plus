/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief
 * @version 0.1
 * @date 2024-04-12
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "CameraParamsConf.hpp"

#include <iostream>

// -----------------------------------
int main(int argc, const char **argv)
{
    std::string       path                    = R"(./config/)";
    std::string       filename                = R"(camera_config.json)";
    CameraParamsConf *cameraParamConfInstance = CameraParamsConf::getInstance(path, filename);

    cameraParamConfInstance->LoadParams();

    auto cameraParams = cameraParamConfInstance->getParams();
    std::cout << "the camera name: " << cameraParams->camera_name << '\n';

    cameraParams->camera_exposure = 42;
    cameraParams->camera_name     = "CCD pole";

    cameraParamConfInstance->SaveParams();

    return 0;
}
