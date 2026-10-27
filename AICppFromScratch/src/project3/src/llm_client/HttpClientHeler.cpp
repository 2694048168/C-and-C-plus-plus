#include "llm_client/HttpClientHeler.h"

#include <exception>

namespace Ithaca {

HttpClientHeler::HttpClientHeler(const LLMBaseConfig &config)
    : mConfig(config)
    , mClient(config.host)
    , mHeaders()
{
    mClient.enable_server_certificate_verification(config.verify_ssl);
    mClient.set_connection_timeout(30);
    mClient.set_read_timeout(30);
    mClient.set_write_timeout(30);

    if ((ApiProvider::DEEPSEEK == config.provider || ApiProvider::OPENAI == config.provider) && !config.api_key.empty())
    {
        mHeaders.emplace("Authorization", "Bearer " + config.api_key);
    }
}

std::string HttpClientHeler::post_request(const std::string &endpoint, const std::string &payload)
{
    auto ret = mClient.Post(endpoint, mHeaders, payload, mConfig.data_type);

    if (!ret)
        throw std::runtime_error("Failed to connect to Local or Cloud API");

    if (200 != ret->status)
        throw std::runtime_error(std::format("HTTP error {} : {}", ret->status, ret->body));

    return ret->body;
}

} // namespace Ithaca
