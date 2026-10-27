#include "ChatClient.h"

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace Ithaca {

ChatClient::ChatClient(const ChatConfig &config)
    : mHttp(config.base)
    , mModel(config.model)
    , mMax_output_tokens(config.max_output_tokens)
{
}

std::string ChatClient::ask(const std::string &prompt, const std::optional<std::string> &instructions,
                            std::optional<int> max_output_tokens)
{
    int tokens = max_output_tokens.value_or(mMax_output_tokens);

    std::string request_body  = build_request(prompt, instructions, tokens);
    std::string response_body = mHttp.post_request(ENDPOINT, request_body);
    return parse_response(response_body);
}

std::string ChatClient::build_request(const std::string &prompt, const std::optional<std::string> &instructions,
                                      int max_output_tokens)
{
    json requests;
    requests["model"]             = mModel;
    requests["input"]             = prompt;
    requests["max_output_tokens"] = max_output_tokens;

    if (instructions.has_value())
        requests["instructions"] = instructions.value();

    return requests.dump();
}

std::string ChatClient::parse_response(const std::string &body)
{
    json response = json::parse(body);

    auto status = response.at("status").get<std::string>();
    if ("failed" == status)
    {
        auto error_code = response.at("error").at("code").get<std::string>();
        auto error_msg  = response.at("error").at("message").get<std::string>();
        throw std::runtime_error(std::format("Ollama API Error {} : {}\n", error_code, error_msg));
    }

    if ("completed" != status)
        throw std::runtime_error(std::format("Unexpected status : {}", status));
    std::cout << "\nResponse received successfully\n";

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

    return complete_text;
}

} // namespace Ithaca