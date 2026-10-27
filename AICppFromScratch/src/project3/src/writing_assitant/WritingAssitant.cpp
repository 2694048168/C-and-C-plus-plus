#include "writing_assitant/WritingAssitant.h"

namespace Ithaca {

WritingAssitant::WritingAssitant(ChatClient &client)
    : mClient(client)
{
    initialize_prompts();
}

void WritingAssitant::initialize_prompts()
{
    mPrompts[PromptType::IMPROVE]
        = "You are a professional writing editor. Improve the following text by:\n"
          "- Fixing grammar and spelling errors\n"
          "- Enhancing clarity and readability\n"
          "- Improving word choice and sentence structure\n"
          "- Maintaining the original meaning and tone\n"
          "Return ONLY the improved text, no explanations or comments.";

    mPrompts[PromptType::EXPAND]
        = "You are a professional writer. Expand the following text to approximately {} words by:\n"
          "- Adding relevant details and examples\n"
          "- Elaborating on key points\n"
          "- Maintaining coherence and flow\n"
          "- Keeping the original tone and style\n"
          "Return ONLY the expanded text, no explanations.";

    mPrompts[PromptType::SUMMARIZE]
        = "You are a professional editor. Summarize the following text to approximately {} words by:\n"
          "- Identifying and keeping  only the most important points\n"
          "- Removing redundant information\n"
          "- Maintaining clarity and coherence\n"
          "- Keeping the original tone and style\n"
          "Return ONLY the summary, no explanations.";

    mPrompts[PromptType::CHANGE_TONE]
        = "You are a professional writer. Rewrite the following text in a {} tone:\n"
          "- Maintain the original meaning and key information\n"
          "- Adjust word choice, sentence structure, and style to match the tone\n"
          "- Keep the content length similar\n"
          "Return ONLY the rewritten text, no explanations.";

    mPrompts[PromptType::TRANSLANE]
        = "You are a professional translator. Translate the following text to {}:\n"
          "- Maintain the original meaning and tone\n"
          "- Use natural, idiomatic expressions in the target language\n"
          "- Preserve formatting if present\n"
          "Return ONLY the translation, no explanations.";
}

std::string WritingAssitant::improve(const std::string &text)
{
    return mClient.ask(text, mPrompts.at(PromptType::IMPROVE));
}

std::string WritingAssitant::expand(const std::string &text, int target_words)
{
    std::string instructions     = std::vformat(mPrompts.at(PromptType::EXPAND), std::make_format_args(target_words));
    int         estimated_tokens = target_words * 2;

    return mClient.ask(text, instructions, estimated_tokens);
}

std::string WritingAssitant::summarize(const std::string &text, int target_words)
{
    std::string instructions = std::vformat(mPrompts.at(PromptType::SUMMARIZE), std::make_format_args(target_words));
    int         estimated_tokens = target_words * 2;

    return mClient.ask(text, instructions, estimated_tokens);
}

std::string WritingAssitant::change_tone(const std::string &text, const std::string &tone)
{
    std::string instructions = std::vformat(mPrompts.at(PromptType::CHANGE_TONE), std::make_format_args(tone));
    return mClient.ask(text, instructions);
}

std::string WritingAssitant::translate(const std::string &text, const std::string &target_language)
{
    std::string instructions = std::vformat(mPrompts.at(PromptType::TRANSLANE), std::make_format_args(target_language));
    return mClient.ask(text, instructions);
}

} // namespace Ithaca