/**
 * @file basic_ollama.hpp
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

#include <exception>
#include <format>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

void BasicOllamaAIExample()
{
    std::cout << "\n---->[Begin] The Ollama location API Example\n\n";

    try
    {
        const std::string question = "Who are you, and Say hello to me in one sentance";

        json request = {
            {            "model", "llama3.2:3b"},
            {            "input",      question},
            {"max_output_tokens",           500}
        };

        // ?just for Debug json content
        // const std::string payload = request.dump(2);
        // std::cout << payload << std::endl;
        const std::string payload = request.dump();

        const std::string url  = "127.0.0.1";
        const int         port = 11434;
        httplib::Client   client(url, port);

        // ?set reasonable timeouts (in seconds)
        client.set_connection_timeout(30);
        client.set_read_timeout(30);
        client.set_write_timeout(30);

        // ?API https headers including Authorzation bearer token
        httplib::Headers  headers{};
        const std::string path      = "/v1/responses";
        const std::string data_type = "application/json";
        const auto        ret       = client.Post(path, headers, payload, data_type);

        if (!ret)
            throw std::runtime_error("Failed to connect to Ollama API");

        if (200 != ret->status)
            throw std::runtime_error(std::format("HTTP error {} : {}", ret->status, ret->body));

        json response = json::parse(ret->body);

        auto status = response.at("status").get<std::string>();
        if ("failed" == status)
        {
            auto error_code = response.at("error").at("code").get<std::string>();
            auto error_msg  = response.at("error").at("message").get<std::string>();
            throw std::runtime_error(std::format("Ollama API Error {} : {}\n", error_code, error_msg));
        }

        if ("completed" != status)
            throw std::runtime_error(std::format("Unexpected status : {}", status));
        std::cout << "Response received successfully\n";
        std::cout << response.dump(2) << "\n";

        // ? parse response JSON format
        std::string complete_text{};

        const auto &output_array = response.at("output");
        for (const auto &item : output_array)
        {
            if ("message" == item.at("type").get<std::string>())
            {
                const auto &content_array = item.at("content");
                for (const auto &content_item : content_array)
                {
                    if ("output_text" == content_item.at("type").get<std::string>())
                        complete_text += content_item.at("text").get<std::string>();
                }
            }
        }

        if (complete_text.empty())
            throw std::runtime_error("No output_text found in response\n");

        std::cout << std::format("The question: {}\n", question);
        std::cout << std::format("The response: {}\n", complete_text);
    }
    catch (const json::exception &ex)
    {
        std::cerr << "JSON exception: " << ex.what() << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
    }

    std::cout << "\n---->[End] The Ollama location API Example\n\n";
}

} // namespace Ithaca