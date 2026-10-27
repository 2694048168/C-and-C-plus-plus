/**
 * @file WritingAssitant.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "llm_client/ChatClient.h"

#include <map>

namespace Ithaca {

// prompt types for different writing operations
enum class PromptType
{
    IMPROVE,
    EXPAND,
    SUMMARIZE,
    CHANGE_TONE,
    TRANSLANE
};

class WritingAssitant
{
private:
    // 依赖注入方式
    ChatClient &mClient;

    std::map<PromptType, std::string> mPrompts;

public:
    // compiler specify member-function
    explicit WritingAssitant(ChatClient &client); // Dependency Injection
    ~WritingAssitant() = default;

    WritingAssitant(const WritingAssitant &other)             = delete;
    WritingAssitant &operator=(const WritingAssitant &other)  = delete;
    WritingAssitant(WritingAssitant &&other)                  = delete;
    WritingAssitant       &operator=(WritingAssitant &&other) = delete;
    WritingAssitant       *operator&() {};
    const WritingAssitant *operator&() const {};

private:
    // Initialize all prompt templates
    void initialize_prompts();

public:
    // improve grammer, clarity, and style
    std::string improve(const std::string &text);

    // expand text to target word count
    std::string expand(const std::string &text, int target_words);

    // summarize text to target word count
    std::string summarize(const std::string &text, int target_words);

    // chnage tone(formal, casual, professional, friendly, etc.)
    std::string change_tone(const std::string &text, const std::string &tone);

    // translate into target language
    std::string translate(const std::string &text, const std::string &target_language);
};

} // namespace Ithaca