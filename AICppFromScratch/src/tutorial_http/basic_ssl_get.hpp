/**
 * @file basic_ssl_get.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-24
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

std::string safe_getenv(const std::string &API_key)
{
    const char *val = std::getenv(API_key.c_str());
    return val ? std::string(val) : std::string();
}

/**
 * @brief 
 * 永久设置用户变量（推荐）：[powershell]
 * [Environment]::SetEnvironmentVariable("API_KEY", "你的密钥内容", "User")
 * 
 * 临时设置（仅用于当前终端测试）：[powershell]
 * $env:API_KEY="你的密钥内容"
 */

void BasicSSLGetExample()
{
    std::cout << "\n---->[Begin] The GET(SSL) method for HTTPS\n\n";

    const std::string api_key_name = "API_KEY";
    const std::string api_key      = safe_getenv(api_key_name);
    if (api_key.empty())
    {
        std::cout << "The API Key not found.\n";
        return;
    }

    const std::string  url = "api.openweathermap.org";
    httplib::SSLClient client(url);

    const std::string path     = "/data/2.5/weather?q=Miami&appid=" + api_key + "&units=imperial";
    const auto        response = client.Get(path);

    // ? 200 is OK, and response is std::shared_ptr<Response>, nullptr is fail
    if (response && 200 == response->status)
    {
        try
        {
            const json result = json::parse(response->body);

            const auto temperature = result.at("main").at("temp").get<double>();
            const auto feels_like  = result.at("main").at("feels_like").get<double>();
            const auto description = result.at("weather").at(0).at("description").get<std::string>();

            std::cout << std::format(
                "Temperature: {:.2f} degress\n"
                "Feels like: {:.2f} degress\n"
                "Conditions: {}\n",
                temperature, feels_like, description);
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
            std::cout << "GET Request failed: no response received\n";
        }
        else
        {
            std::cout << std::format("GET Request failed: HTTPS {}\n", response->status);
        }
    }

    std::cout << "\n---->[Een] The GET(SSL) method for HTTPS\n\n";
}

} // namespace Ithaca
