/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Bitcoin exchange rates
 * @version 0.1
 * @date 2024-01-30
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "curl_easy.h"
#include "curl_exception.h"
#include "json.hpp"

#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <string_view>

using json = nlohmann::json;

/**
 * @brief Bitcoin exchange rates
 * 
 * Write a program that displays Bitcoin exchange rates for the most important
 * currencies (such as USD, EUR, or GBP). The exchange rates must be fetched from
 *  an online service, such as: https://blockchain.info.
 * 
 */

/**
 * @brief Solution: Asio C++ Library
 https://think-async.com/Asio/
 https://github.com/chriskohlhoff/asio/
------------------------------------------------------ */
struct exchange_info
{
    double      delay_15m_price;
    double      latest_price;
    double      buying_price;
    double      selling_price;
    std::string symbol;
};

using blockchain_rates = std::map<std::string, exchange_info>;

void from_json(const json &jdata, exchange_info &info)
{
    info.delay_15m_price = jdata.at("15m").get<double>();
    info.latest_price    = jdata.at("last").get<double>();
    info.buying_price    = jdata.at("buy").get<double>();
    info.selling_price   = jdata.at("sell").get<double>();
    info.symbol          = jdata.at("symbol").get<std::string>();
}

std::stringstream get_json_document(std::string_view url)
{
    std::stringstream str;

    try
    {
        curl::curl_ios<std::stringstream> writer(str);
        curl::curl_easy                   easy(writer);

        easy.add<CURLOPT_URL>(url.data());
        easy.add<CURLOPT_FOLLOWLOCATION>(1L);

        easy.perform();
    }
    catch (const curl::curl_easy_exception &error)
    {
        auto errors = error.get_traceback();
        error.print_traceback();
    }

    return str;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto doc = get_json_document("https://blockchain.info/ticker");

    json jdata;
    doc >> jdata;

    blockchain_rates rates = jdata;

    for (const auto &kvp : rates)
    {
        std::cout << "1BPI = " << kvp.second.latest_price << " " << kvp.first << std::endl;
    }

    return 0;
}
