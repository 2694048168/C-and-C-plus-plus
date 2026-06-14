/**
 * @file logger.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Logger utility for application logging
 * @version 0.1
 * @date 2026-06-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>

namespace Ithaca::Logger {
inline void log(const std::string &message, const std::string &filename = "log.txt")
{
    std::ofstream file(filename, std::ios::app);

    if (file.is_open())
    {
        file << message << std::endl;
        file.close();
        std::cout << message << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not open log file " << filename << std::endl;
    }
}

} // namespace Ithaca::Logger
