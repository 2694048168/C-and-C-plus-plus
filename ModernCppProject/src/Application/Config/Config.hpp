/**
 * @file Config.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-04-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <string>

struct AppConfig
{
    std::string appName       = "Ithaca_Project";
    std::string configVersion = "V1.0.0";
};

class ApplicationConfigParam
{
public:
    static ApplicationConfigParam &getInstance();

private:
    // 单例
    ApplicationConfigParam();
    ~ApplicationConfigParam();
    ApplicationConfigParam(const ApplicationConfigParam &)            = delete;
    ApplicationConfigParam &operator=(const ApplicationConfigParam &) = delete;

private:
    AppConfig mParam;
};
