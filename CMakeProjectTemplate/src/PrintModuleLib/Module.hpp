/**
 * @file Module.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 使用单例模式设计一个打印消息的模块
 * @version 0.1
 * @date 2024-04-04
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <string>

class PrintModule
{
public:
    PrintModule(const PrintModule &t)            = delete;
    PrintModule &operator=(const PrintModule &t) = delete;

    static PrintModule *getInstance()
    {
        return m_pInstance;
    }

    void printDebug(const std::string &message);
    void printDebug(const std::string &message, const int& val);
    
    void printInfo(const std::string &message);
    void printInfo(const std::string &message, const int& val);

    void printError(const std::string &message);
    void printError(const std::string &message, const int& val);

private:
    PrintModule() = default;
    static PrintModule *m_pInstance;
};
