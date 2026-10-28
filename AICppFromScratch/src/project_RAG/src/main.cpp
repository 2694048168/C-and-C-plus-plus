/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "RAG/EmbeddingClient.h"
#include "RAG/RAGClient.h"

#include <Windows.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>

std::string safe_getenv(const std::string &API_key)
{
    const char *val = std::getenv(API_key.c_str());
    return val ? std::string(val) : std::string();
}

Ithaca::EmbeddingConfig create_config(Ithaca::ApiProvider provider)
{
    if (Ithaca::ApiProvider::OPENAI == provider || Ithaca::ApiProvider::DEEPSEEK == provider)
    {
        return Ithaca::EmbeddingConfig{
            .ai_base{.provider = provider, .api_key = safe_getenv("API_KEY"), .host = "https://api.openai.com"},
            .vector_store_host = "http://localhost:8080",
            .project_name      = "Ithaca_rag_demo",
            .model_name        = "text-embedding-3-small"
        };
    }
    else // local ollama
    {
        return Ithaca::EmbeddingConfig{
            .ai_base{.provider = Ithaca::ApiProvider::OLLAMA, .api_key = "", .host = "http://127.0.0.1:11434"},
            .vector_store_host = "http://localhost:8080",
            .project_name      = "Ithaca_rag_demo",
            .model_name        = "nomic-embed-text"
        };
    }
}

Ithaca::ChatConfig create_chat_config(Ithaca::ApiProvider provider)
{
    if (Ithaca::ApiProvider::OPENAI == provider || Ithaca::ApiProvider::DEEPSEEK == provider)
    {
        return Ithaca::ChatConfig{
            .base{.provider = provider, .api_key = safe_getenv("API_KEY"), .host = "https://api.openai.com"},
            .model             = "gpt-4o-mini", // deepseek-v4-flash
            .max_output_tokens = 500
        };
    }
    else // local ollama
    {
        return Ithaca::ChatConfig{
            .base{.provider = Ithaca::ApiProvider::OLLAMA, .api_key = "", .host = "http://127.0.0.1:11434"},
            .model             = "llama3.2",
            .max_output_tokens = 500
        };
    }
}

int main(int argc, const char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "\n------------------------------------>";
    std::cout << "\n<-------- RAG Demo -------->";
    std::cout << "\n------------------------------------>\n";

    const std::string data_path = R"(./data/)";

    try
    {
        // auto config = create_config(Ithaca::ApiProvider::DEEPSEEK);
        // auto config = create_config(Ithaca::ApiProvider::OPENAI);
        auto config      = create_config(Ithaca::ApiProvider::OLLAMA);
        auto chat_config = create_chat_config(Ithaca::ApiProvider::OLLAMA);

        Ithaca::EmbeddingClient embed_client(config);
        Ithaca::ChatClient      chat_client(chat_config);

        Ithaca::RAGClient rag_system(embed_client, chat_client, config.project_name, config.model_name, false);

        std::cout << "cleaning vector store for " << config.project_name << "...";
        if (embed_client.clear())
            std::cout << "Done...\n";

        if (!std::filesystem::exists(data_path))
        {
            std::cout << std::format("Error: Data folder: {} NOT found\n", data_path);
            return -1;
        }

        std::cout << "Scanning " << data_path << " for text files...\n";
        for (const auto &entry : std::filesystem::directory_iterator(data_path))
        {
            if (".txt" == entry.path().extension())
            {
                std::ifstream ifs(entry.path());
                if (!ifs.is_open())
                {
                    std::cerr << "Counld not open: " << entry.path().filename() << "\n";
                    continue;
                }

                std::cout << "Ingesting: " << entry.path().filename() << "\n";

                std::string line{};
                int         line_count = 0;
                while (std::getline(ifs, line))
                {
                    if (line.empty() || line.find_first_not_of(" \t\n\r") == std::string::npos)
                        continue;

                    std::cout << ". ";
                    std::vector<float> embedding_vector = embed_client.create_embedding(line);

                    if (embed_client.upsert(line, embedding_vector))
                        ++line_count;
                }

                std::cout << "\n ---> successfully stored " << line_count << " lines.\n";
            }
        }

        std::cout << "\nIngestion Complete.\n";
        std::cout << "Check the Dashboard at http://localhost:8080\n";

        std::cout << "\n\n===================================\n";
        std::cout << "    RAG Query system - Interactive\n";
        std::cout << "===================================\n";
        std::cout << "Ask question about your knowledge base\n";
        std::cout << "Type 'exit' to quit\n";
        std::cout << "===================================\n";

        std::string input{};
        while (true)
        {
            std::cout << "\nYou: ";
            std::getline(std::cin, input);

            if (input.empty())
                continue;

            if ("exit" == input || "quit" == input)
                break;

            std::string answer = rag_system.ask(input);
            std::cout << "\nAssitant: " << answer << "\n";
        }

        std::cout << "\nGoodbye!\n";
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
        return -1;
    }

    return 0;
}
