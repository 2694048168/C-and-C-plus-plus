/**
 * @file test_datetime_main.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-04-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include "VersionDefined.h"

#include <iostream>
#include <string>

// ---------------------------------------------------------
namespace {
const int         major      = 1;
const int         minor      = 2;
const int         patch      = 3;
const std::string branch     = "main";
const std::string build_date = __DATE__ " " __TIME__;
} // namespace

std::string getVersionString()
{
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

std::string getBuildInfo()
{
    return "Branch: " + branch + ", Build: " + build_date;
}

// ------------------------------------
int main(int argc, const char *argv[])
{
    // C/C++编译器会内置有两个获取编译时间的宏：__DATE__和__TIME__;
    std::cout << "Date: " << __DATE__ << '\n';
    std::cout << "Time: " << __TIME__ << '\n';

    std::cout << "The Compiler Version: \n";
    std::cout << std::string(COMPLETE_VERSION) << '\n';

    std::cout << "The Current Version: \t" << getVersionString() << '\n';
    std::cout << "The Current Build: \t" << getBuildInfo() << '\n';

    return 0;
}
