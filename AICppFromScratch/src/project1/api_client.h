/**
 * @file api_client.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "api_config.h"
#include "httplib.h"
#include "nlohmann/json.hpp"

#include <exception>
#include <format>
#include <iostream>

using json = nlohmann::json;

namespace Ithaca {

std::string safe_getenv(const std::string &API_key);

std::string build_request(const ApiConfig &config, const std::string &prompt);

httplib::Headers build_headers(const ApiConfig &config);

std::string parse_response_body(const std::string &body);

std::string make_api_call(httplib::Client &client, const ApiConfig &config, const std::string &prompt);

} // namespace Ithaca