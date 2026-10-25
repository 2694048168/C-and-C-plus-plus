/**
 * @file robust_ssl.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <format>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

std::string get_http_status_message(int status_code)
{
    switch (status_code)
    {
    case 200:
        return "OK";
        break;
    case 201:
        return "Created";
        break;
    case 204:
        return "No Content";
        break;
    case 400:
        return "Bad Request";
        break;
    case 401:
        return "Unauthorized";
        break;
    case 403:
        return "Forbidden";
        break;
    case 404:
        return "Not Found";
        break;
    case 429:
        return "Too Many Requests";
        break;
    case 500:
        return "Internal Server Error";
        break;
    case 502:
        return "Bad Gateway";
        break;
    case 503:
        return "Service Unavilable";
        break;
    case 504:
        return "Gateway Timeout";
        break;
    default:
        return "Unknow Status Code";
        break;
    }
}

void BasicRobustSSLExample()
{
    std::cout << "\n---->[Begin] The Robust POST(SSL) method for HTTPS\n\n";

    const json post_data = {
        { "title",                        "Modern AI and C++"},
        {  "body", "This is a tutorial for AI via Modern C++"},
        {"userId",                                       2851},
    };

    const std::string  url = "jsonplaceholder.typicode.com";
    httplib::SSLClient client(url);

    // !enable verification for production security
    // ? set true for production
    client.enable_server_certificate_verification(false);

    // ?set reasonable timeouts (in seconds)
    client.set_connection_timeout(10);
    client.set_read_timeout(10);
    client.set_write_timeout(10);

    // ?set default headers that will be send with every request
    client.set_default_headers({
        {"User-Agent", "CppApp/1.0"},
    });

    const std::string path      = "/posts";
    const std::string data_type = "application/json";
    const auto        response  = client.Post(path, post_data.dump(), data_type);

    // ? 201 is OK by Created, and response is std::shared_ptr<Response>, nullptr is fail
    if (response && 201 == response->status)
    {
        try
        {
            const json result = json::parse(response->body);

            const auto id     = result.at("id").get<int>();
            const auto title  = result.at("title").get<std::string>();
            const auto body   = result.at("body").get<std::string>();
            const auto userId = result.at("userId").get<int>();

            std::cout << std::format("POST ID created by Server: {}\nTitle: {}\nBody: {}\nuserID: {}", id, title, body,
                                     userId);
        }
        catch (const json::exception &ex)
        {
            std::cerr << "JSON exception: " << ex.what() << '\n';
        }
        catch (const std::exception &ex)
        {
            std::cerr << "C++ std exception: " << ex.what() << '\n';
        }
    }
    else
    {
        if (!response)
        {
            std::cout << "POST Request failed: no response received\n";
        }
        else
        {
            std::cout << std::format("POST Request failed: HTTPS {}-{}\n", response->status,
                                     get_http_status_message(response->status));
        }
    }

    std::cout << "\n---->[End] The Robust POST(SSL) method for HTTPS\n\n";
}

} // namespace Ithaca