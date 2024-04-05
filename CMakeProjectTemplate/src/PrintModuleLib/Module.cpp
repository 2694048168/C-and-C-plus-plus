#include "Module.hpp"

#include <iostream>

// 饿汉模式, 定义类的时候就创建(new)该单例对象
// 没有多线程安全问题
PrintModule *PrintModule::m_pInstance = new PrintModule;

void PrintModule::printDebug(const std::string &message)
{
    std::cout << "[====Debug====]" + message << std::endl;
}

void PrintModule::printDebug(const std::string &message, const int &val)
{
    std::cout << "[====Debug====]" + message << val << std::endl;
}

void PrintModule::printInfo(const std::string &message)
{
    std::cout << "[====Info====]" + message << std::endl;
}

void PrintModule::printInfo(const std::string &message, const int &val)
{
    std::cout << "[====Info====]" + message << val << std::endl;
}

void PrintModule::printError(const std::string &message)
{
    std::cout << "[====Error====]" + message << std::endl;
}

void PrintModule::printError(const std::string &message, const int &val)
{
    std::cout << "[====Error====]" + message << val << std::endl;
}
