/**
 * @file RAGClient.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "RAG/EmbeddingClient.h"
#include "llm_client/ChatClient.h"

namespace Ithaca {

class RAGClient
{
private:
    EmbeddingClient &mEmbedClient;
    ChatClient      &mChatClient;

    std::string mProjectName;
    std::string mModelName;
    bool        mDebugMode{false};

public:
    std::string ask(const std::string &user_question);

public:
    // compiler specify member-function
    explicit RAGClient(EmbeddingClient &ec, ChatClient &cc, const std::string &project, const std::string &model,
                       bool debug_mode = false);
    ~RAGClient() = default;

    RAGClient(const RAGClient &other)             = delete;
    RAGClient &operator=(const RAGClient &other)  = delete;
    RAGClient(RAGClient &&other)                  = delete;
    RAGClient       &operator=(RAGClient &&other) = delete;
    RAGClient       *operator&() {};
    const RAGClient *operator&() const {};
};

} // namespace Ithaca
