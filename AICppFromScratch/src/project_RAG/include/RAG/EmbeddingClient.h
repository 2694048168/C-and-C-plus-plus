/**
 * @file EmbeddingClient.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "RAG/Embedding.h"
#include "llm_client/HttpClientHeler.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

using json = nlohmann::json;

namespace Ithaca {

class EmbeddingClient
{
private:
    HttpClientHeler mAiHttp;
    HttpClientHeler mVectorHttp;

    std::string mModelName;
    std::string mProjectName;

public:
    bool clear();

    std::vector<float> create_embedding(const std::string &text);

    bool upsert(const std::string &text, const std::vector<float> &vector_embed);

    std::vector<SearchResult> serarch(const std::vector<float> &query_vector, int top_k = 3);

public:
    // compiler specify member-function
    explicit EmbeddingClient(const EmbeddingConfig &config);
    ~EmbeddingClient() = default;

    EmbeddingClient(const EmbeddingClient &other)             = delete;
    EmbeddingClient &operator=(const EmbeddingClient &other)  = delete;
    EmbeddingClient(EmbeddingClient &&other)                  = delete;
    EmbeddingClient       &operator=(EmbeddingClient &&other) = delete;
    EmbeddingClient       *operator&() {};
    const EmbeddingClient *operator&() const {};
};

} // namespace Ithaca