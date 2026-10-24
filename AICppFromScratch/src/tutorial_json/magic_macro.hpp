/**
 * @file magic_macro.hpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-24
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "nlohmann/json.hpp"

#include <format>
#include <iomanip>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace Ithaca {

struct CameraInfo
{
    std::string camName    = "HIK";
    std::string camType    = "line-scan";
    int         resolution = 8192;
    bool        isColor    = false;
    double      gain       = 3.24;
};

// ?regirster Macor to_json and from_json automatically
// !Macro throws exception so need handling
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraInfo, camName, camType, resolution, isColor, gain)

void JsonMacroExample()
{
    std::cout << "---->[Begin] The JSON magic macro Example\n\n";

    // ?struct into json
    CameraInfo camInfo;
    json       camJson = camInfo;
    std::cout << "Struct --> JSON:\n" << camJson.dump() << "\n";

    // ?json into struct
    std::string raw_str = R"(
        {
            "camName":"INS",
            "camType":"area",
            "resolution": 16384,
            "isColor":true,
            "gain":2.2
        })";

    CameraInfo camInfo2 = json::parse(raw_str);
    std::cout << "Struct --> JSON:\n";
    std::cout << camInfo2.camName << "\n";
    std::cout << camInfo2.camType << "\n";
    std::cout << camInfo2.resolution << "\n";

    std::cout << "---->[end] The JSON magic macro Example\n\n";
}
} // namespace Ithaca