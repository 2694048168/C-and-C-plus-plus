/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "api_client.h"

#include <iostream>

int main(int argc, const char *argv[])
{
    std::cout << "\n----------------------------------------------------->";
    std::cout << "\n---->OpenAI/DeepSeek Cloud or Ollama Local Execution";
    std::cout << "\n----------------------------------------------------->\n";

    try
    {
        Ithaca::ApiConfig config{.provider = Ithaca::ApiProvider::OLLAMA,
                                 .model    = "llama3.2:3b",
                                 .host     = "http://127.0.0.1:11434"};

        if ((Ithaca::ApiProvider::DEEPSEEK == config.provider || Ithaca::ApiProvider::OPENAI == config.provider)
            && config.api_key.empty())
        {
            throw std::runtime_error("API_KEY get unsuceesffully");
        }

        httplib::Client client(config.host);
        client.enable_server_certificate_verification(false);
        client.set_connection_timeout(30);
        client.set_read_timeout(30);
        client.set_write_timeout(30);

        std::string prompt        = "Who are you, and Say hello to me in one sentance";
        auto        complete_text = Ithaca::make_api_call(client, config, prompt);
        std::cout << std::format("The Prompt: {}\n", prompt);
        std::cout << std::format("The Response: {}\n", complete_text);

        prompt        = "What is 100 + 42 - 12";
        complete_text = Ithaca::make_api_call(client, config, prompt);
        std::cout << std::format("The Prompt: {}\n", prompt);
        std::cout << std::format("The Response: {}\n", complete_text);
    }
    catch (const json::exception &ex)
    {
        std::cerr << "JSON exception: " << ex.what() << '\n';
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
    }

    return 0;
}
