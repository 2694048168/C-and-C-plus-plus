/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Finding the IP address of a host
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#define ASIO_STANDALONE
#include "asio/asio.hpp"

/**
 * @brief Finding the IP address of a host
 * 
 * Write a program that can retrieve and print the IPv4 address of a host.
 * If multiple addresses are found, then all of them should be printed. 
 * The program should work on all platforms.
 * 
 */

/**
 * @brief Solution: Asio C++ Library
 https://think-async.com/Asio/
 https://github.com/chriskohlhoff/asio/
------------------------------------------------------ */
std::vector<std::string> get_ip_address(std::string_view hostname)
{
    std::vector<std::string> ips;

    try
    {
        asio::io_context        context;
        asio::ip::tcp::resolver resolver(context);
        auto                    endpoints = resolver.resolve(asio::ip::tcp::v4(), hostname.data(), "");

        for (auto e = endpoints.begin(); e != endpoints.end(); ++e)
            ips.push_back(((asio::ip::tcp::endpoint)*e).address().to_string());
    }
    catch (const std::exception &e)
    {
        std::cerr << "exception: " << e.what() << std::endl;
    }

    return ips;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto ips = get_ip_address("packtpub.com");

    for (const auto &ip : ips)
    {
        std::cout << ip << std::endl;
    }

    return 0;
}
