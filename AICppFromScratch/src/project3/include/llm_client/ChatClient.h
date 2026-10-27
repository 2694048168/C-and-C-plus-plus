/**
 * @file ChatClient.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "HttpClientHeler.h"

#include <optional>

namespace Ithaca {

struct ChatConfig
{
    LLMBaseConfig base{};
    std::string   model{"llama3.2:3b"};
    int           max_output_tokens{500};
};

class ChatClient
{
private:
    HttpClientHeler mHttp;
    std::string     mModel;
    int             mMax_output_tokens;

    static constexpr const char *ENDPOINT = "/v1/responses";

public:
    std::string ask(const std::string &prompt, const std::optional<std::string> &instructions = std::nullopt,
                    std::optional<int> max_output_token = std::nullopt);

public:
    // compiler specify member-function
    explicit ChatClient(const ChatConfig &config);
    ~ChatClient() = default;

    ChatClient(const ChatClient &other)             = delete;
    ChatClient &operator=(const ChatClient &other)  = delete;
    ChatClient(ChatClient &&other)                  = delete;
    ChatClient       &operator=(ChatClient &&other) = delete;
    ChatClient       *operator&() {};
    const ChatClient *operator&() const {};

private:
    std::string build_request(const std::string &prompt, const std::optional<std::string> &instructions,
                              int max_output_token);

    std::string parse_response(const std::string &body);
};

} // namespace Ithaca