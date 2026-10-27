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

#include "llm_client/ChatClient.h"
#include "writing_assitant/WritingAssitant.h"

#include <format>
#include <iostream>

#include <Windows.h>

int main(int argc, const char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "\n------------------------------------------->";
    std::cout << "\n<-------- AI Writing Assitant -------->";
    std::cout << "\n------------------------------------------->\n";

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

        Ithaca::WritingAssitant assitant(chat);

        std::string original_text    = "me and my friend went to store yesterday. we buyed some foods and drinks.";
        auto        improved_text    = assitant.improve(original_text);
        auto        expanded_text    = assitant.expand(improved_text, 150);
        auto        change_tone_text = assitant.change_tone(improved_text, "formal");
        auto        summary_text     = assitant.summarize(improved_text, 25);
        auto        spanish_text     = assitant.translate(improved_text, "Spanish");
        auto        chinese_text     = assitant.translate(improved_text, "Chinese");

        std::cout << std::format("The original_text: {}\n\n", original_text);
        std::cout << std::format("The improved_text: {}\n\n", improved_text);
        std::cout << std::format("The expanded_text(150): {}\n\n", expanded_text);
        std::cout << std::format("The change_tone_text: {}\n\n", change_tone_text);
        std::cout << std::format("The summary_text: {}\n\n", summary_text);
        std::cout << std::format("The spanish_text: {}\n\n", spanish_text);
        std::cout << std::format("The chinese_text: {}\n\n", chinese_text);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
    }

    return 0;
}
