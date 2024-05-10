/**
 * @file 06_regularExpressions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-05-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cstdio>
#include <regex>
#include <string>

/**
 * @brief 正则表达式(regular expression)是定义搜索模式的字符串,
 * *正则表达式在计算机科学中有着悠久的历史, 形成了一种用于搜索、替换和提取语言数据的迷你语言.
 * STL 在＜regex＞头文件中提供了正则表达式支持.
 * 
 * ====使用称为模式(pattern)的字符串建立正则表达式,
 * 模式使用特定的正则表达式语法来代表所需的字符串集,
 * 换句话说, 模式定义了感兴趣的所有可能字符串的子集.
 * 
 * ====修改后的 ECMAScript 正则表达式语法
 * 1. 字符类 Character Classes
 * 2. 量词 Quantifiers
 * 3. 分组 Groups
 * 4. 其他特殊字符 Other Special Characters
 * 
 * TODO: Localizations 本地化
 * std::locale 是一个用于编码文化偏好的类, 它通常在应用程序运行的操作环境中进行编码,
 * 它还控制了许多首选项, 例如字符串比较, 日期和时间, 货币和数字格式, 邮政编码, 电话号码.
 * STL 在＜locale＞头文件中提供了 std::locale 类和许多帮助函数.
 * 
 */

// -----------------------------------
int main(int argc, const char **argv)
{
    printf("\nstd::basic_regex constructs from a string literal\n");
    std::regex zip_regex{R"((\w{2})?(\d{5})(-\d{4})?)"};

    assert(zip_regex.mark_count() == 3);

    printf("\nstd::sub_match\n");
    std::regex  regex_{R"((\w{2})(\d{5})(-\d{4})?)"};
    std::smatch results;

    // returns true given matching string
    {
        std::string zip("NJ07936-3173");
        const auto  matched = std::regex_match(zip, results, regex_);
        assert(matched);
        assert(results[0] == "NJ07936-3173");
        assert(results[1] == "NJ");
        assert(results[2] == "07936");
        assert(results[3] == "-3173");
    }

    // returns false given non-matching string
    {
        std::string zip("Iomega Zip 100");
        const auto  matched = std::regex_match(zip, results, regex_);
        assert(!matched);
    }

    printf("\nwhen only part of a string matches a regex, std::regex_ \n");
    {
        std::regex  regex{R"((\w{2})(\d{5})(-\d{4})?)"};
        std::string sentence("The string NJ07936-3173 is a ZIP Code.");
        // match returns false
        assert(!std::regex_match(sentence, regex));

        // search returns true
        assert(std::regex_search(sentence, regex));
    }

    {
        printf("\nstd::regex_replace\n");
        std::regex  regex{"[aeoiu]"};
        std::string phrase("queueing and cooeeing in eutopia");
        const auto  result = std::regex_replace(phrase, regex, "_");
        assert(result == "q_____ng _nd c_____ng _n __t_p__");
    }

    return 0;
}
