/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Translating text to any language
 * @version 0.1
 * @date 2024-01-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "curl_easy.h"
#include "curl_exception.h"
#include "curl_header.h"

#include <codecvt>
#include <iostream>
#include <locale>
#include <memory>
#include <regex>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

/**
 * @brief Translating text to any language
 * 
 * Write a program that can translate text from one language to another 
 * using an online service. It should be possible to specify the text that 
 * you wish to translate, the language of the text, and the language to translate to.
 * 
 */

/**
 * @brief Solution: Asio C++ Library
 https://think-async.com/Asio/
 https://github.com/chriskohlhoff/asio/
------------------------------------------------------ */
std::wstring utf8_to_utf16(std::string_view text)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring                                           wtext = converter.from_bytes(text.data());
    return wtext;
}

std::string utf16_to_utf8(std::wstring_view wtext)
{
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::string                                            text = converter.to_bytes(wtext.data());
    return text;
}

class text_translator
{
public:
    text_translator(std::string_view endpoint, std::string_view key)
        : endpoint(endpoint)
        , app_key(key)
    {
    }

    std::wstring translate_text(std::wstring_view wtext, std::string_view to, std::string_view from = "en")
    {
        try
        {
            using namespace std::string_literals;

            std::stringstream str;
            std::string       text = utf16_to_utf8(wtext);

            curl::curl_ios<std::stringstream> writer(str);
            curl::curl_easy                   easy(writer);

            curl::curl_header header;
            header.add("Ocp-Apim-Subscription-Key:" + app_key);

            easy.escape(text);
            auto url = endpoint + "/Translate";
            url += "?from="s + from.data();
            url += "&to="s + to.data();
            url += "&text="s + text;

            easy.add<CURLOPT_URL>(url.c_str());
            easy.add<CURLOPT_HTTPHEADER>(header.get());

            easy.perform();

            auto result = deserialize_result(str.str());
            return utf8_to_utf16(result);
        }
        catch (const curl::curl_easy_exception &error)
        {
            auto errors = error.get_traceback();
            error.print_traceback();
        }
        catch (const std::exception &ex)
        {
            std::cout << ex.what() << std::endl;
        }

        return {};
    }

private:
    std::string deserialize_result(std::string_view text)
    {
        std::regex  rx(R"(<string.*>(.*)<\/string>)");
        std::cmatch match;
        if (std::regex_search(text.data(), match, rx))
        {
            return match[1];
        }

        return "";
    }

    std::string endpoint;
    std::string app_key;
};

void set_utf8_conversion(std::wostream &stream)
{
    auto        codecvt = std::make_unique<std::codecvt_utf8<wchar_t>>();
    std::locale utf8locale(std::locale(), codecvt.get());
    codecvt.release();
    stream.imbue(utf8locale);
}

// ------------------------------
int main(int argc, char **argv)
{
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    set_utf8_conversion(std::wcout);

    text_translator tt("https://api.microsofttranslator.com/V2/Http.svc", "...(your api key)...");

    std::vector<std::tuple<std::wstring, std::string, std::string>> texts{
        {       L"hello world!", "en", "ro"},
        {   L"what time is it?", "en", "es"},
        {L"ceci est un exemple", "fr", "en"}
    };

    for (const auto [text, from, to] : texts)
    {
        auto result = tt.translate_text(text, to, from);

        std::cout << from << ": ";
        std::wcout << text << std::endl;
        std::cout << to << ": ";
        std::wcout << result << std::endl;
    }

    return 0;
}
