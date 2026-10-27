/**
 * @file api_config.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <string>

namespace Ithaca {

enum class ApiProvider
{
    OPENAI,
    OLLAMA,
    DEEPSEEK
};

struct ApiConfig
{
    ApiProvider provider{ApiProvider::OLLAMA};
    std::string model{"llama3.2:3b"};
    std::string api_key{}; // empty for ollama

    // full url for client
    std::string host{"http://127.0.0.1:11434"};
    std::string endpoint{"/v1/responses"};
    std::string data_type{"application/json"};
    int         max_output_token{500};
};

} // namespace Ithaca