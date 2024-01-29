/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Computing file hashes
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1

#include "files.h"
#include "hex.h"
#include "md5.h"
#include "sha.h"

namespace fs = std::filesystem;

/**
 * @brief Computing file hashes
 * 
 * Write a program that, given a path to a file, computes and prints to the console
 * the SHA1, SHA256, and MD5 hash values for the content of the file.
 * 
 */

/**
 * @brief Solution: 
------------------------------------------------------ */
template<class Hash>
std::string compute_hash(const fs::path &filepath)
{
    std::ifstream file(filepath.string(), std::ios::binary);
    if (file.is_open())
    {
        Hash           hash;
        CryptoPP::byte digest[Hash::DIGESTSIZE] = {0};

        do
        {
            char buffer[4096] = {0};
            file.read(buffer, 4096);

            auto extracted = static_cast<size_t>(file.gcount());

            if (extracted > 0)
            {
                hash.Update(reinterpret_cast<CryptoPP::byte *>(buffer), extracted);
            }
        }
        while (!file.fail());

        hash.Final(digest);

        CryptoPP::HexEncoder encoder;
        std::string          result;

        encoder.Attach(new CryptoPP::StringSink(result));
        encoder.Put(digest, sizeof(digest));
        encoder.MessageEnd();

        return result;
    }

    throw std::runtime_error("Cannot open file!");
}

template<class Hash>
std::string compute_hash_ex(const fs::path &filepath)
{
    std::string digest;
    Hash        hash;

    CryptoPP::FileSource source(
        filepath.c_str(), true,
        new CryptoPP::HashFilter(hash, new CryptoPP::HexEncoder(new CryptoPP::StringSink(digest))));

    return digest;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::string path;
    std::cout << "Path: ";
    std::cin >> path;

    try
    {
        std::cout << "SHA1: " << compute_hash<CryptoPP::SHA1>(path) << std::endl;
        std::cout << "SHA256: " << compute_hash<CryptoPP::SHA256>(path) << std::endl;
        std::cout << "MD5: " << compute_hash<CryptoPP::Weak::MD5>(path) << std::endl;

        std::cout << "SHA1: " << compute_hash_ex<CryptoPP::SHA1>(path) << std::endl;
        std::cout << "SHA256: " << compute_hash_ex<CryptoPP::SHA256>(path) << std::endl;
        std::cout << "MD5: " << compute_hash_ex<CryptoPP::Weak::MD5>(path) << std::endl;
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << std::endl;
    }

    return 0;
}
