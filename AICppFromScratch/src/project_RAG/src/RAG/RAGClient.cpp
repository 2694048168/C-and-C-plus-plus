#include "RAG/RAGClient.h"

namespace Ithaca {

RAGClient::RAGClient(EmbeddingClient &ec, ChatClient &cc, const std::string &project, const std::string &model,
                     bool debug_mode)
    : mEmbedClient(ec)
    , mChatClient(cc)
    , mProjectName(project)
    , mModelName(model)
    , mDebugMode(debug_mode)
{
}

std::string RAGClient::ask(const std::string &user_question)
{
    // Step 1. Retrieve: Convert question into vector and find top matches
    auto query_vec     = mEmbedClient.create_embedding(user_question);
    auto retrieved_ret = mEmbedClient.serarch(query_vec, 3);

    if (retrieved_ret.empty())
    {
        if (mDebugMode)
        {
            std::cout << "\n[Debug] No results found in vector store!\n";
            std::cout << "Possible causes:\n";
            std::cout << " - VectorStoreServer has no data for the project/model\n";
            std::cout << " - Project name or model name mismatch\n";
            std::cout << " - Ingestor hasn't been run yet\n";
            std::cout << "-------------------------------------------------\n";
        }

        return "I don't have any information in my knowledge base.\nPlease ensure the knowledge base has been "
               "populated.\n";
    }

    if (mDebugMode)
    {
        std::cout << "\n[Debug] Retriever Results (Ordered by Similarity):\n";
        for (const auto &res : retrieved_ret)
        {
            std::printf(" ---> Score: %.4f | Text: %s\n", res.score, res.text.c_str());
        }
        std::cout << "-------------------------------------------------\n";
    }

    // ? construct strict instructions and content
    std::stringstream ss{};
    ss << "You are a grounded technical assitant that answers questions using retrieved knowledge.\n\n"
       << "STRICT RULES:\n"
       << "- Use ONLY the information from the retrieved facts below.\n"
       << "- Do NOT use outside knowledge, guesses, or assumptions\n"
       << "- Do NOT expand, rephrase, or add details that are not explicitly stated in the retrieved facts.\n"
       << "- If a definition appears incomplete, leave it incomplete rather than adding outside information.\n"
       << "- Every statement in the answer must be directly supported by at least one retrieved fact.\n"
       << "- The retrieved facts are sentence-level fragments; no single fact is guaranteed to be complete.\n"
       << "- You MUST fuse related facts into one coherent answer when possible.\n"
       << "- Ignore facts that are irrelevent to the question.\n\n"
       << "FUSION INSTRUCTIONS:\n"
       << "1. Read all facts completely.\n"
       << "2. Identify the score question intent.\n"
       << "3. Extract only the parts that directly supported the inent.\n"
       << "4. Combine those parts into a single explanation.\n"
       << "5. Prefer higher-relevance facts if overlap exists.\n\n"
       << "OUTPUT REQUIREMENRS:\n"
       << "- Write one cohesive answer.\n"
       << "- Do NOT mention retrieval or relevance scores.\n\n"
       << "FAILURE CONDITION:\n"
       << "If the facts do not contain enough information, respond EXACTLY with:\n"
       << "'I don't have enough information to answer that question.'\n\n"
       << "RETRIEVED FACTS (ranked by relevance):\n";

    for (const auto &res : retrieved_ret)
    {
        ss << "- [Relevance: " << std::fixed << std::setprecision(2) << res.score << "] " << res.text << "\n";
    }

    return mChatClient.ask(user_question, ss.str(), 500);
}

} // namespace Ithaca