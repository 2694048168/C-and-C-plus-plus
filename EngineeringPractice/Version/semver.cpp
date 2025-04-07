#include "semver.h"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <vector>

namespace Version {
std::vector<std::string> split(const std::string &s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string              token;
    std::istringstream       tokenStream(s);
    while (getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

bool isNumeric(const std::string &s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), std::isdigit);
}

int compareIdentifiers(const std::string &a, const std::string &b)
{
    bool aNumeric = isNumeric(a);
    bool bNumeric = isNumeric(b);

    if (aNumeric && bNumeric)
    {
        int numA = stoi(a);
        int numB = stoi(b);
        return numA - numB;
    }
    else if (aNumeric)
    {
        return -1; // 数字标识符比非数字标识符优先级低
    }
    else if (bNumeric)
    {
        return 1;
    }
    else
    {
        return a.compare(b);
    }
}

void SemVer::validate() const
{
    if (major_ < 0 || minor_ < 0 || patch_ < 0)
    {
        throw std::invalid_argument("Version components cannot be negative");
    }
}

SemVer SemVer::fromString(const std::string &versionStr)
{
    size_t plusPos   = versionStr.find('+');
    size_t hyphenPos = versionStr.find('-');

    std::string coreVersion;
    std::string prerelease;
    std::string build;

    // 分离核心版本、预发布和构建元数据
    if (plusPos != std::string::npos)
    {
        build       = versionStr.substr(plusPos + 1);
        coreVersion = versionStr.substr(0, plusPos);
    }
    else
    {
        coreVersion = versionStr;
    }

    if (hyphenPos != std::string::npos && (plusPos == std::string::npos || hyphenPos < plusPos))
    {
        prerelease  = coreVersion.substr(hyphenPos + 1,
                                         (plusPos == std::string::npos ? std::string::npos : plusPos - hyphenPos - 1));
        coreVersion = coreVersion.substr(0, hyphenPos);
    }

    // 解析核心版本
    std::vector<std::string> parts = split(coreVersion, '.');
    if (parts.size() != 3)
    {
        throw std::invalid_argument("Invalid version format");
    }

    try
    {
        int major = stoi(parts[0]);
        int minor = stoi(parts[1]);
        int patch = stoi(parts[2]);

        return SemVer(major, minor, patch, prerelease, build);
    }
    catch (const std::exception &)
    {
        throw std::invalid_argument("Invalid version number");
    }
}

std::string SemVer::toString() const
{
    std::stringstream ss;
    ss << major_ << "." << minor_ << "." << patch_;

    if (!prerelease_.empty())
    {
        ss << "-" << prerelease_;
    }

    if (!build_.empty())
    {
        ss << "+" << build_;
    }

    return ss.str();
}

bool SemVer::operator==(const SemVer &other) const
{
    return major_ == other.major_ && minor_ == other.minor_ && patch_ == other.patch_
        && prerelease_ == other.prerelease_;
}

bool SemVer::operator<(const SemVer &other) const
{
    if (major_ != other.major_)
        return major_ < other.major_;
    if (minor_ != other.minor_)
        return minor_ < other.minor_;
    if (patch_ != other.patch_)
        return patch_ < other.patch_;

    // 预发布版本比较
    if (prerelease_.empty() && !other.prerelease_.empty())
        return false;
    if (!prerelease_.empty() && other.prerelease_.empty())
        return true;
    if (prerelease_.empty() && other.prerelease_.empty())
        return false;

    std::vector<std::string> thisPre  = split(prerelease_, '.');
    std::vector<std::string> otherPre = split(other.prerelease_, '.');

    size_t maxSize = std::max(thisPre.size(), otherPre.size());
    for (size_t i = 0; i < maxSize; ++i)
    {
        if (i >= thisPre.size())
            return true;
        if (i >= otherPre.size())
            return false;

        int cmp = compareIdentifiers(thisPre[i], otherPre[i]);
        if (cmp != 0)
            return cmp < 0;
    }

    return false;
}

// 其他比较操作符基于==和<实现
bool SemVer::operator!=(const SemVer &other) const
{
    return !(*this == other);
}

bool SemVer::operator>(const SemVer &other) const
{
    return other < *this;
}

bool SemVer::operator<=(const SemVer &other) const
{
    return !(other < *this);
}

bool SemVer::operator>=(const SemVer &other) const
{
    return !(*this < other);
}

} // namespace Version
