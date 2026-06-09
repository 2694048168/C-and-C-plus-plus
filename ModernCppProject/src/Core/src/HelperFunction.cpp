#include "Core/HelperFunction.hpp"

#include <ctime>
#include <format>
#include <memory>
#include <sstream>

namespace IthacaCore {

const std::string GetCurrentTimestamp() noexcept
{
    time_t    now = time(0);
    struct tm timeStruct;
    char      timeStrBuf[80];
    localtime_s(&timeStruct, &now);
    strftime(timeStrBuf, sizeof(timeStrBuf), "%Y%m%d%H%M%S", &timeStruct);
    return timeStrBuf;
}

} // namespace IthacaCore