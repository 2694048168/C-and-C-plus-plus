/**
 * @file HelperFunction.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief Helper functions for string conversion and timestamp generation
 * @version 0.1
 * @date 2026-04-09
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "Core/SymbolExport.hpp"

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace IthacaCore {

/**
* @brief 获取系统当前时间戳,
* 时间格式: 年月日时分秒, 如 20241225140223
* Get current date/time, format is YYYYMMDDHHmmss
*
* @return 返回时间戳格式化后的字符串 const std::string
*/
[[nodiscard]] LIB_API const std::string CALLING_CONVENTIONS GetCurrentTimestamp() noexcept;

} // namespace IthacaCore
