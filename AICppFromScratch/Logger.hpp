/**
 * @file Logger.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Logger utility for application logging
 * @version 0.1
 * @date 2026-06-07
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <fstream>
#include <iostream>
#include <string>

namespace Ithaca {
class Logger
{
public:
    void log(const std::string &message, const std::string &filename = "log.txt")
    {
        std::ofstream file(filename, std::ios::app);

        if (file.is_open())
        {
            file << message << std::endl;
            file.close();
        }
        else
        {
            std::cerr << "Error: Could not open log file " << filename << std::endl;
        }
    }
};

} // namespace Ithaca
