/**
 * @file HttpClientHeler.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "httplib.h"

#include <string>

namespace Ithaca {

enum class ApiProvider
{
    OPENAI,
    OLLAMA,
    DEEPSEEK
};

struct LLMBaseConfig
{
    ApiProvider provider{ApiProvider::OLLAMA};
    std::string api_key{};
    std::string host{"http://127.0.0.1:11434"};
    std::string data_type{"application/json"};

    bool verify_ssl{false};
    int  connection_timeout{30};
    int  read_timeout{30};
    int  write_timeout{30};
};

class HttpClientHeler
{
private:
    LLMBaseConfig    mConfig;
    httplib::Client  mClient;
    httplib::Headers mHeaders;

public:
    std::string post_request(const std::string &endpoint, const std::string &payload);

public:
    // compiler specify member-function
    explicit HttpClientHeler(const LLMBaseConfig &config);
    ~HttpClientHeler() = default;

    HttpClientHeler(const HttpClientHeler &other)             = delete;
    HttpClientHeler &operator=(const HttpClientHeler &other)  = delete;
    HttpClientHeler(HttpClientHeler &&other)                  = delete;
    HttpClientHeler       &operator=(HttpClientHeler &&other) = delete;
    HttpClientHeler       *operator&() {};
    const HttpClientHeler *operator&() const {};
};

} // namespace Ithaca