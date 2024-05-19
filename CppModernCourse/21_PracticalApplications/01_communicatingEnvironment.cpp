/**
 * @file 01_communicatingEnvironment.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdlib>
#include <iostream>
#include <string>

/**
 * @brief 与环境交互
 * 可能想创建另一个进程, 例如谷歌的 Chrome 浏览器启动了许多进程来服务一个浏览器会话,
 * 通过搭载操作系统的进程模型, 这建立了一些安全性和稳健性, 例如Web 应用程序和插件在单独的进程中运行,
 * 因此如果它们崩溃, 整个浏览器不会崩溃; 此外通过在单独的进程中运行浏览器的渲染引擎,安全漏洞将变得更加难以利用,
 * 因为Google 在所谓的沙盒环境中锁定了该进程的权限.
 * 
 */

// ----------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief std::system
     * 使用＜cstdlib＞头文件中的 std::system 函数启动一个单独的进程,
     * 该函数接受与要执行的命令相对应的 C-风格字符串, 并返回与命令的返回码相对应的 int;
     * *具体的行为取决于操作环境, 例如, 在 Windows 机器上该函数将调用 cmd.exe;
     * *在 Linux 机器上将调用 /bin/sh. 该函数在命令仍在执行时阻塞.
     * 
     */
    std::cout << "[====]std::system\n";

#ifdef _WIN32
    std::string command{"ping google.com"};
#elif __linux__
    std::string command{"ping -c 4 google.com"};
#elif __APPLE__
    std::string command{"ping -c 4 google.com"};
#endif

    const auto result = std::system(command.c_str());
    std::cout << "The command \'" << command << "\' returned " << result << "\n";

    /**
    * @brief std::getenv
    * *操作环境通常有环境变量, 用户和开发者可以设置这些变量, 以帮助程序找到程序运行所需的重要信息.
    * ＜cstdlib＞头文件包含 std::getenv 函数, 它接受一个与想要查找的环境变量名称相对应的 C-风格字符串,
    * 并返回一个包含相应变量内容的 C-风格字符串; 如果没有找到这样的变量,该函数将返回 nullptr;
    */
    std::cout << "[====]std::getenv\n";
    std::string variable_name{"PATH"};
    std::string res{std::getenv(variable_name.c_str())};
    std::cout << "The variable " << variable_name << " equals " << res << "\n";

    std::string gcc_variable_name{"VULKAN_SDK"};
    std::string res_{std::getenv(variable_name.c_str())};
    std::cout << "The variable " << gcc_variable_name << " equals " << res_ << "\n";

    return 0;
}
