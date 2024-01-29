/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Validating user credentials
 * @version 0.1
 * @date 2024-01-29
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "hex.h"
#include "sha.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief Validating user credentials
 * 
 * Write a program that simulates the way users authenticate to a secured system. 
 * In order to log in, a user must be already registered with the system. 
 * The user enters a username and a password and the program checks 
 * if it matches any of its registered users; if it does, the user is granted access,
 * otherwise, the operation fails. For security reasons, the system must not record
 * the password but use an SHA hash instead.
 * 
 */

/**
 * @brief Solution: Crypto++® Library 8.9
 https://cryptopp.com/
------------------------------------------------------ */
struct User
{
    int         id;
    std::string username;
    std::string password;
    std::string firstname;
    std::string lastname;
};

std::string get_hash(std::string_view password)
{
    CryptoPP::SHA512 sha;
    CryptoPP::byte   digest[CryptoPP::SHA512::DIGESTSIZE];

    sha.CalculateDigest(digest, reinterpret_cast<const CryptoPP::byte *>(password.data()), password.length());

    CryptoPP::HexEncoder encoder;
    std::string          result;

    encoder.Attach(new CryptoPP::StringSink(result));
    encoder.Put(digest, sizeof(digest));
    encoder.MessageEnd();

    return result;
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<User> users{
        {101,  "scarface",
         "07A8D53ADAB635ADDF39BAEACFB799FD7C5BFDEE365F3AA721B7E25B54A4E87D419ADDEA34BC3073BAC472DCF4657E50C0F6781DDD8FE"
         "883653D10F7930E78FF",   "Tony",  "Montana"},
        {202,       "neo",
         "C2CC277BCC10888ECEE90F0F09EE9666199C2699922EFB41EA7E88067B2C075F3DD3FBF3CFE9D0EC6173668DD83C111342F91E941A2CA"
         "DC46A3A814848AA9B05", "Thomas", "Anderson"},
        {303, "godfather",
         "0EA7A0306FE00CD22DF1B835796EC32ACC702208E0B052B15F9393BCCF5EE9ECD8BAAF27840D4D3E6BCC3BB3B009259F6F73CC77480C0"
         "65DDE67CD9BEA14AA4D",   "Vito", "Corleone"}
    };

    std::string username, password;
    std::cout << "Username: ";
    std::cin >> username;

    std::cout << "Password: ";
    std::cin >> password;

    auto hash = get_hash(password);

    auto pos = std::find_if(std::begin(users), std::end(users),
                            [username, hash](const user &u) { return u.username == username && u.password == hash; });

    if (pos != std::end(users))
        std::cout << "Login successful!" << std::endl;
    else
        std::cout << "Invalid username or password" << std::endl;

    return 0;
}
