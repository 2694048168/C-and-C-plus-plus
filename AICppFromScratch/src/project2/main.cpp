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

#include "ChatClient.h"

#include <format>
#include <iostream>

int main(int argc, const char *argv[])
{
    std::cout << "\n----------------------------------------------------->";
    std::cout << "\n---->OpenAI/DeepSeek Cloud or Ollama Local OOP";
    std::cout << "\n----------------------------------------------------->\n";

    try
    {
        Ithaca::ChatConfig config{
            .base  = {.provider = Ithaca::ApiProvider::OLLAMA, .api_key = "", .host = "http://127.0.0.1:11434"},
            .model = "llama3.2:3b",
            .max_output_tokens = 500
        };

        if ((Ithaca::ApiProvider::DEEPSEEK == config.base.provider
             || Ithaca::ApiProvider::OPENAI == config.base.provider)
            && config.base.api_key.empty())
        {
            throw std::runtime_error("API_KEY get unsuceesffully");
        }

        Ithaca::ChatClient chat(config);

        std::string prompt      = "Say hello to me in one sentance";
        auto        ai_response = chat.ask(prompt);
        std::cout << std::format("The Prompt: {}\n", prompt);
        std::cout << std::format("The Response: {}\n", ai_response);

        prompt                                 = "What is Modern C++?";
        std::optional<std::string> instruction = "Talk like a C++ tutor.";
        ai_response                            = chat.ask(prompt, instruction);
        std::cout << std::format("The Prompt: {}\n", prompt);
        std::cout << std::format("The Response: {}\n", ai_response);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
    }

    return 0;
}
