/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cassert>
#include <ctime>
#include <iostream>
#include <regex>
#include <string>

#ifdef USE_BOOST_OPTIONAL
#    include <boost/optional.hpp>
using boost::optional;
#else
#    include <optional>
using std::optional;
#endif

/**
 * @brief Extracting URL parts
 *  Write a function that, given a string that represents a URL,
 * parses and extracts the parts of the URL
 * (protocol, domain, port, path, query, and fragment).
 */

/**
 * @brief Solution:

 This problem is also suited to being solved using regular expressions. Finding a regular
expression that could match any URL is, however, a difficult task. The purpose of this
exercise is to help you practice your skills with the regex library, and not to find the
ultimate regular expression for this particular purpose. Therefore, the regular expression
used here is provided only for didactic purposes.

# regular expressions using online testers and debuggers
https://regex101.com/
---------------------------------------------- */
struct uri_parts
{
    std::string           protocol;
    std::string           domain;
    optional<int>         port;
    optional<std::string> path;
    optional<std::string> query;
    optional<std::string> fragment;
};

optional<uri_parts> parse_uri(std::string uri)
{
    std::regex rx(R"(^(\w+):\/\/([\w.-]+)(:(\d+))?([\w\/\.]+)?(\?([\w=&]*)(#?(\w+))?)?$)");
    auto       matches = std::smatch{};

    if (std::regex_match(uri, matches, rx))
    {
        if (matches[1].matched && matches[2].matched)
        {
            uri_parts parts;
            parts.protocol = matches[1].str();
            parts.domain   = matches[2].str();
            if (matches[4].matched)
                parts.port = std::stoi(matches[4]);
            if (matches[5].matched)
                parts.path = matches[5];
            if (matches[7].matched)
                parts.query = matches[7];
            if (matches[9].matched)
                parts.fragment = matches[9];

            return parts;
        }
    }

    return {};
}

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string currentDateTime()
{
    time_t    now = time(0);
    struct tm tstruct;
    char      buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

    return buf;
}

// --------------------------------
int main(int argc, char **argv)
{
    auto p1 = parse_uri("https://baidu.com");
    assert(p1);
    assert(p1->protocol == "https");
    assert(p1->domain == "baidu.com");
    assert(!p1->port);
    assert(!p1->path);
    assert(!p1->query);
    assert(!p1->fragment);

    auto p2 = parse_uri("https://bbc.com:80/en/index.html?lite=true#ui");
    assert(p2);
    assert(p2->protocol == "https");
    assert(p2->domain == "bbc.com");
    assert(p2->port == 80);
    assert(p2->path.value() == "/en/index.html");
    assert(p2->query.value() == "lite=true");
    assert(p2->fragment.value() == "ui");

    std::cout << "[" << currentDateTime() << "] All test thought successfully\n";

    return 0;
}
