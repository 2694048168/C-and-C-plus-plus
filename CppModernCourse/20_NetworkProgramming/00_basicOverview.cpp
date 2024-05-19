/**
 * @file 00_basicOverview.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-19
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    /**
     * @brief 用 Boost Asio 进行网络编程
     * Windows 操作系统上使用 WinSocket 进行网络TCP/IP编程;
     * Linux 操作系统上使用 socket API 进行网络TCP/IP编程;
     */
    std::cout << "用 Boost Asio 进行网络编程\n";
    std::cout << "Windows 操作系统上使用 WinSocket 进行网络TCP/IP编程\n";
    std::cout << "Linux 操作系统上使用 socket API 进行网络TCP/IP编程\n";

    return 0;
}
