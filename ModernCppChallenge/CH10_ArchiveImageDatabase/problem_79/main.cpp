/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Finding files in a ZIP archive
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ziplib/ZipArchive.h"
#include "ziplib/ZipFile.h"

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Finding files in a ZIP archive
 * 
 * Write a program that can search for and print all the files in a ZIP archive
 *  whose name matches a user-provided regular expression 
 * (for instance, use ^.*\.jpg$ to find all files with the extension .jpg).
 * 
 */

/**
 * @brief Solution: zip-library in Modern C++11
 https://bitbucket.org/wbenny/ziplib/src/master/
------------------------------------------------------ */
std::vector<std::string> find_in_archive(const std::filesystem::path &archivePath, std::string_view pattern)
{
    std::vector<std::string> results;

    if (std::filesystem::exists(archivePath))
    {
        try
        {
            auto archive = ZipFile::Open(archivepath.string());

            for (size_t i = 0; i < archive->GetEntriesCount(); ++i)
            {
                auto entry = archive->GetEntry(i);
                if (entry)
                {
                    if (!entry->IsDirectory())
                    {
                        auto name = entry->GetName();
                        if (std::regex_match(name, std::regex{pattern.data()}))
                        {
                            results.push_back(entry->GetFullName());
                        }
                    }
                }
            }
        }
        catch (const std::exception &ex)
        {
            std::cout << ex.what() << std::endl;
        }
    }

    return results;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::string archive_path = "./sample79.zip";
    std::filesystem::path archivepath(archive_path);
    std::string pattern     = "^.*\.jpg";

    std::cout << "Results:" << std::endl;

    for (const auto &name : find_in_archive(archivepath, pattern))
    {
        std::cout << name << std::endl;
    }

    return 0;
}
