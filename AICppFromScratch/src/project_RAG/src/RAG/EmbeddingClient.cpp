#include "RAG/EmbeddingClient.h"

namespace Ithaca {

EmbeddingClient::EmbeddingClient(const EmbeddingConfig &config)
    : mAiHttp(config.ai_base)
    , mVectorHttp(LLMBaseConfig{.host = config.vector_store_host})
    , mModelName(config.model_name)
    , mProjectName(config.project_name)
{
}

bool EmbeddingClient::clear()
{
    std::string response
        = mVectorHttp.post_request("v1/vectors/clear", json(ClearRequest{mModelName, mProjectName}).dump());

    ClearResponse clear_response = json::parse(response);

    return clear_response.status == "success";
}

std::vector<float> EmbeddingClient::create_embedding(const std::string &text)
{
    json body{};
    body["input"] = text;
    body["model"] = mModelName;

    std::string response = mAiHttp.post_request("v1/embeddings", body.dump());

    EmbeddingResponse embedding_response = json::parse(response);

    if (embedding_response.data.empty())
        throw std::runtime_error("No embedding data returned from API");

    return embedding_response.data[0].embedding;
}

bool EmbeddingClient::upsert(const std::string &text, const std::vector<float> &vector_embed)
{
    std::string response = mVectorHttp.post_request(
        "v1/vectors/upsert", json(UpsertRequest{text, vector_embed, mModelName, mProjectName}).dump());

    UpsertResponse upsert_response = json::parse(response);

    return upsert_response.status == "success";
}

std::vector<SearchResult> EmbeddingClient::serarch(const std::vector<float> &query_vector, int top_k)
{
    std::string response = mVectorHttp.post_request(
        "v1/vectors/query", json(SearchRequest{query_vector, mModelName, mProjectName, top_k}).dump());

    SearchResponse search_response = json::parse(response);

    return search_response.matches;
}

} // namespace Ithaca