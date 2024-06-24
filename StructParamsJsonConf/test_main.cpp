/**
 * @file test_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief ini config file via C++ in project
 * @version 0.1
 * @date 2024-06-24
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "inicpp.hpp"

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    // =========== save ini conf-file ===========
    std::string path = "config/ini_conf.ini";

    ini::IniFile ini_conf;

    ini_conf["Foo"]["hello"]    = "world";
    ini_conf["Foo"]["float"]    = 1.02f;
    ini_conf["Foo"]["int"]      = 123;
    ini_conf["Another"]["char"] = 'q';
    ini_conf["Another"]["bool"] = true;

    ini_conf.save(path);

    std::cout << "Saved ini file.\n";

    // =========== load ini conf-file ===========
    ini::IniFile ini_file;
    ini_file.load(path);

    // show the parsed contents of the ini file
    std::cout << "Parsed ini-configure file contents\n";
    std::cout << "Has " << ini_file.size() << " sections\n";
    for (const auto &sectionPair : ini_file)
    {
        const std::string     &sectionName = sectionPair.first;
        const ini::IniSection &section     = sectionPair.second;
        std::cout << "Section '" << sectionName << "' has " << section.size() << " fields" << std::endl;

        for (const auto &fieldPair : sectionPair.second)
        {
            const std::string   &fieldName = fieldPair.first;
            const ini::IniField &field     = fieldPair.second;
            std::cout << "  Field '" << fieldName << "' Value '" << field.as<std::string>() << "'" << std::endl;
        }
    }

    return 0;
}
