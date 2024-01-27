/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Compressing and decompressing files to/from a ZIP archive with a password
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "ziplib/ZipArchive.h"
#include "ziplib/ZipFile.h"
#include "ziplib/utils/stream_utils.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

/**
 * @brief Compressing and decompressing files to/from a ZIP archive with a password
 * 
 * Write a program that can do the following:
 * 1. Compress either a file or the contents of a user-specified directory, recursively,
 *  to a password-protected ZIP archive
 * 2. Decompress the content of a password-protected ZIP archive to a user-specified destination directory
 * 
 */

/**
 * @brief Solution: zip-library in Modern C++11
 https://bitbucket.org/wbenny/ziplib/src/master/
------------------------------------------------------ */
void compress(const fs::path &source, const fs::path &archive, std::string_view password,
              std::function<void(std::string_view)> reporter)
{
    if (fs::is_regular_file(source))
    {
        if (reporter)
            reporter("Compressing " + source.string());
        ZipFile::AddEncryptedFile(archive.string(), source.string(), source.filename().string(), password.data(),
                                  LzmaMethod::Create());
    }
    else
    {
        for (const auto &p : fs::recursive_directory_iterator(source))
        {
            if (reporter)
                reporter("Compressing " + p.path().string());

            if (fs::is_directory(p))
            {
                auto zipArchive = ZipFile::Open(archive.string());
                auto entry      = zipArchive->CreateEntry(p.path().string());
                entry->SetAttributes(ZipArchiveEntry::Attributes::Directory);
                ZipFile::SaveAndClose(zipArchive, archive.string());
            }
            else if (fs::is_regular_file(p))
            {
                ZipFile::AddEncryptedFile(archive.string(), p.path().string(), p.path().filename().string(),
                                          password.data(), LzmaMethod::Create());
            }
        }
    }
}

void ensure_directory_exists(const fs::path &dir)
{
    if (!fs::exists(dir))
    {
#ifdef USE_BOOST_FILESYSTEM
        boost::system::error_code err;
#else
        std::error_code err;
#endif
        fs::create_directories(dir, err);
    }
}

void decompress(const fs::path &destination, const fs::path &archive, std::string_view password,
                std::function<void(std::string_view)> reporter)
{
    ensure_directory_exists(destination);

    auto zipArchive = ZipFile::Open(archive.string());

    for (size_t i = 0; i < zipArchive->GetEntriesCount(); ++i)
    {
        auto entry = zipArchive->GetEntry(i);
        if (entry)
        {
            auto filepath = destination / fs::path{entry->GetFullName()}.relative_path();
            if (reporter)
                reporter("Creating " + filepath.string());

            if (entry->IsPasswordProtected())
                entry->SetPassword(password.data());

            if (entry->IsDirectory())
            {
                ensure_directory_exists(filepath);
            }
            else
            {
                ensure_directory_exists(filepath.parent_path());

                std::ofstream destFile;
                destFile.open(filepath.string().c_str(), std::ios::binary | std::ios::trunc);

                if (!destFile.is_open())
                {
                    if (reporter)
                        reporter("Cannot create destination file!");
                }

                auto dataStream = entry->GetDecompressionStream();
                if (dataStream)
                {
                    utils::stream::copy(*dataStream, destFile);
                }
            }
        }
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    char option = 0;
    std::cout << "Select [c]ompress/[d]ecompress?";
    std::cin >> option;

    if (option == 'c')
    {
        std::string archivepath;
        std::string inputpath;
        std::string password;
        std::cout << "Enter file or dir to compress:";
        std::cin >> inputpath;
        std::cout << "Enter archive path:";
        std::cin >> archivepath;
        std::cout << "Enter password:";
        std::cin >> password;

        compress(inputpath, archivepath, password, [](std::string_view message) { std::cout << message << std::endl; });
    }
    else if (option == 'd')
    {
        std::string archivepath;
        std::string outputpath;
        std::string password;
        std::cout << "Enter dir to decompress:";
        std::cin >> outputpath;
        std::cout << "Enter archive path:";
        std::cin >> archivepath;
        std::cout << "Enter password:";
        std::cin >> password;

        decompress(outputpath, archivepath, password,
                   [](std::string_view message) { std::cout << message << std::endl; });
    }
    else
    {
        std::cout << "invalid option" << std::endl;
    }

    return 0;
}
