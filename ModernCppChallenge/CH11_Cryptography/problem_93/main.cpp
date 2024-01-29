/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Encrypting and decrypting files
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "default.h"
#include "files.h"
#include "hex.h"
#include "sha.h"

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

namespace fs = std::filesystem;

/**
 * @brief Encrypting and decrypting files
 * 
 * Write a program that can encrypt and decrypt files using 
 * the Advanced Encryption Standard (AES or Rijndael). It should be possible to
 *  specify both a source file and a destination file path, as well as a password.
 * 
 */

/**
 * @brief Solution: 
------------------------------------------------------ */
void encrypt_file(const fs::path &sourcefile, const fs::path &destfile, std::string_view password)
{
    CryptoPP::FileSource source(
        sourcefile.c_str(), true,
        new CryptoPP::DefaultEncryptorWithMAC((CryptoPP::byte *)password.data(), password.size(),
                                              new CryptoPP::FileSink(destfile.c_str())));
}

void encrypt_file(const fs::path &filepath, std::string_view password)
{
    auto temppath = fs::temp_directory_path() / filepath.filename();

    encrypt_file(filepath, temppath, password);

    fs::remove(filepath);
    fs::rename(temppath, filepath);
}

void decrypt_file(const fs::path &sourcefile, const fs::path &destfile, std::string_view password)
{
    CryptoPP::FileSource source(
        sourcefile.c_str(), true,
        new CryptoPP::DefaultDecryptorWithMAC((CryptoPP::byte *)password.data(), password.size(),
                                              new CryptoPP::FileSink(destfile.c_str())));
}

void decrypt_file(const fs::path &filepath, std::string_view password)
{
    auto temppath = fs::temp_directory_path() / filepath.filename();

    decrypt_file(filepath, temppath, password);

    fs::remove(filepath);
    fs::rename(temppath, filepath);
}

// ------------------------------
int main(int argc, char **argv)
{
    encrypt_file("sample.txt", "sample.txt.enc", "cppchallenger");
    decrypt_file("sample.txt.enc", "sample.txt.dec", "cppchallenger");

    encrypt_file("sample.txt", "cppchallenger");
    decrypt_file("sample.txt", "cppchallenger");

    return 0;
}
