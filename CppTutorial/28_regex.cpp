/**
 * @file 28_regex.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 现代C++ 正则表达式
 * @version 0.1
 * @date 2026-10-14
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include <fstream>
#include <iostream>
#include <regex>
#include <string>

void analyzeLogFile(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        return;

    std::regex timePattern(R"((\d{2}:\d{2}:\d{2}))");
    std::regex ipPattern(R"((\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))");
    std::regex errorPattern(R"(error|ERROR)");

    std::string line;
    while (std::getline(file, line))
    {
        std::smatch timeMatch, ipMatch, errorMatch;
        if (std::regex_search(line, timeMatch, timePattern) && std::regex_search(line, ipMatch, ipPattern)
            && std::regex_search(line, errorMatch, errorPattern))
        {
            std::cout << "Timestamp: " << timeMatch[0] << ", IP: " << ipMatch[0] << ", Error: " << errorMatch[0]
                      << std::endl;
        }
    }
}

// ====================================
int main(int argc, const char *argv[])
{
    const std::string filename = "log/log-2026-1014.log";
    analyzeLogFile(filename);

    return 0;
}

// clang++ 28_regex.cpp -std=c++23 -o demo.exe
// g++ 28_regex.cpp -std=c++23 -o demo.exe
