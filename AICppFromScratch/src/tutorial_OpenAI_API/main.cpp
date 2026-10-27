/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-25
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "basic_ollama.hpp"
#include "basic_openai.hpp"

#include <iostream>

int main(int argc, const char *argv[])
{
    std::cout << "\n---->[Begin] The Tutorial for OpenAI API\n\n";

    Ithaca::BasicOpenAIExample();
    Ithaca::BasicOllamaAIExample();

    std::cout << "\n---->[End] The Tutorial for OpenAI API\n\n";

    return 0;
}
