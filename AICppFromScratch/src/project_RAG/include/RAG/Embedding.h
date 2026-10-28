/**
 * @file Embedding.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "llm_client/ChatClient.h"
#include "nlohmann/json.hpp"

#include <string>
#include <vector>

namespace Ithaca {

struct EmbeddingConfig
{
    LLMBaseConfig ai_base{};
    std::string   vector_store_host{};
    std::string   project_name{};
    std::string   model_name{};
};

struct ClearRequest
{
    std::string model{};
    std::string project{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ClearRequest, model, project)

struct UpsertRequest
{
    std::string        text{};
    std::vector<float> vector_embed{};
    std::string        model{};
    std::string        project{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UpsertRequest, text, vector_embed, model, project)

struct SearchRequest
{
    std::vector<float> vector_embed{};
    std::string        model{};
    std::string        project{};
    int                top_k{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchRequest, vector_embed, model, project, top_k)

struct ClearResponse
{
    std::string status{};
    int         rows_deleted{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ClearResponse, status, rows_deleted)

struct UpsertResponse
{
    std::string status{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(UpsertResponse, status)

struct EmbeddingData
{
    std::vector<float> embedding{};
    int                index{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EmbeddingData, embedding, index)

struct EmbeddingResponse
{
    std::vector<EmbeddingData> data{};
    std::string                model{};
    std::string                object{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(EmbeddingResponse, data, model, object)

struct SearchResult
{
    std::string text{};
    float       score{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchResult, text, score)

struct SearchResponse
{
    std::vector<SearchResult> matches{};
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SearchResponse, matches)

} // namespace Ithaca