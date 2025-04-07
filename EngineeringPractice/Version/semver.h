/**
 * @file semver.h
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-04-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once
#include <stdexcept>
#include <string>

namespace Version {

class SemVer
{
public:
    // 构造函数
    SemVer(int major, int minor, int patch, const std::string &prerelease = "", const std::string &build = "")
        : major_(major)
        , minor_(minor)
        , patch_(patch)
        , prerelease_(prerelease)
        , build_(build)
    {
        validate();
    }

    // 从字符串解析
    static SemVer fromString(const std::string &versionStr);

    // 转换为字符串
    std::string toString() const;

    // 比较操作符
    bool operator==(const SemVer &other) const;
    bool operator!=(const SemVer &other) const;
    bool operator<(const SemVer &other) const;
    bool operator>(const SemVer &other) const;
    bool operator<=(const SemVer &other) const;
    bool operator>=(const SemVer &other) const;

    // Getters
    int getMajor() const
    {
        return major_;
    }

    int getMinor() const
    {
        return minor_;
    }

    int getPatch() const
    {
        return patch_;
    }

    std::string getPrerelease() const
    {
        return prerelease_;
    }

    std::string getBuild() const
    {
        return build_;
    }

private:
    void validate() const;

    int         major_;
    int         minor_;
    int         patch_;
    std::string prerelease_; // 预发布版本标识符，如"alpha.1"
    std::string build_;      // 构建元数据，如"exp.sha.5114f85"
};

} // namespace Version
