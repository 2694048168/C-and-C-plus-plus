/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-24
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "basic_get.hpp"
#include "basic_post.hpp"
#include "basic_ssl_get.hpp"
#include "basic_ssl_post.hpp"
#include "robust_ssl.hpp"

#include <iostream>

int main(int argc, const char *argv[])
{
    std::cout << "\n---->[Begin] The Tutorial for HTTP\n\n";

    Ithaca::BasicGetExample();
    Ithaca::BasicPostExample();
    Ithaca::BasicSSLGetExample();
    Ithaca::BasicSSLPostExample();
    Ithaca::BasicRobustSSLExample();

    std::cout << "\n---->[End] The Tutorial for HTTP\n\n";

    return 0;
}
