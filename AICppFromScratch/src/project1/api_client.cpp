#include "api_client.h"

namespace Ithaca {

std::string safe_getenv(const std::string &API_key)
{
    const char *val = std::getenv(API_key.c_str());
    return val ? std::string(val) : std::string();
}

std::string build_request(const ApiConfig &config, const std::string &prompt)
{
    json request = {
        {            "model",            config.model},
        {            "input",                  prompt},
        {"max_output_tokens", config.max_output_token}
    };

    return request.dump();
}

httplib::Headers build_headers(const ApiConfig &config)
{
    httplib::Headers headers{};
    if (Ithaca::ApiProvider::OLLAMA != config.provider)
        headers = {
            {"Authorization", "Bearer " + config.api_key}
        };

    return headers;
}

std::string parse_response_body(const std::string &body)
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

std::string make_api_call(httplib::Client &client, const ApiConfig &config, const std::string &prompt)
{
    const std::string payload = build_request(config, prompt);

    auto headers = build_headers(config);

    const auto ret = client.Post(config.endpoint, headers, payload, config.data_type);
    if (!ret)
        throw std::runtime_error("Failed to connect to Local or Cloud API");

    if (200 != ret->status)
        throw std::runtime_error(std::format("HTTP error {} : {}", ret->status, ret->body));

    return parse_response_body(ret->body);
}

} // namespace Ithaca