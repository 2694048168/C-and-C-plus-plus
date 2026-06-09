/**
 * @file Application.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-04-11
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Config/Config.hpp"
#include "Core/HelperFunction.hpp"
#include "Logger/Logger.hpp"

class Application
{
public:
    void Run();

public:
    // 单例模式
    static Application &getInstance();
    Application(const Application &)            = delete;
    Application &operator=(const Application &) = delete;

private:
    Application();
    ~Application();
};