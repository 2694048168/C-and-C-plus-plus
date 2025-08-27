/**
 * @file ParameterManager.hpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 参数管理 模板类
 * @version 0.1
 * @date 2025-08-27
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <any>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <vector>

class ParameterManager
{
public:
    template<typename T>
    void SetParam(const std::string &key, T &&value)
    {
        // 检查是否为嵌套键(包含点号)
        auto dot_pos = key.find('.');
        if (std::string::npos == dot_pos)
        {
            // 普通键, 直接设置
            mParams[key] = std::forward<T>(value);
        }
        else
        {
            // 嵌套键, 需要递归处理
            std::vector<std::string> keys;
            size_t                   start = 0;
            while (dot_pos != std::string::npos)
            {
                keys.emplace_back(key.substr(start, dot_pos - start));
                start   = dot_pos + 1;
                dot_pos = key.find('.', start);
            }
            keys.emplace_back(key.substr(start));

            set_nested_param_impl(key, std::forward<T>(value), keys, 0);
        }
    }

    // 获取参数(无默认值版本, 失败时抛出异常)
    template<typename T>
    T GetParam(const std::string &key) const
    {
        // 检查是否为嵌套键
        auto dot_pos = key.find('.');
        if (std::string::npos == dot_pos)
        {
            // 普通键
            auto iter = mParams.find(key);
            if (mParams.end() == iter)
            {
                // logging
                throw std::runtime_error("Key '" + key + "' not found");
            }

            try
            {
                return std::any_cast<T>(iter->second);
            }
            catch (const std::bad_any_cast &ex)
            {
                // logging
                throw std::runtime_error("Type mismatch for key '" + key + "'");
            }
        }
        else
        {
            // 嵌套键, 递归获取
            std::vector<std::string> keys;
            size_t                   start = 0;
            while (dot_pos != std::string::npos)
            {
                keys.emplace_back(key.substr(start, dot_pos - start));
                start   = dot_pos + 1;
                dot_pos = key.find('.', start);
            }
            keys.emplace_back(key.substr(start));

            const ParameterManager *pCurrent = this;
            for (size_t idx{0}; idx < keys.size() - 1; ++idx)
            {
                auto iter = pCurrent->mParams.find(keys[idx]);
                if (pCurrent->mParams.end() == iter)
                {
                    // logging
                    throw std::runtime_error("Key '" + keys[idx] + "' not found in nested dictionary");
                }

                try
                {
                    pCurrent = &std::any_cast<const ParameterManager &>(iter->second);
                }
                catch (const std::bad_any_cast &ex)
                {
                    // logging
                    throw std::runtime_error("Key '" + keys[idx] + "' is not a nested dictionary");
                }
            }

            // 最后一级键
            auto iter = pCurrent->mParams.find(keys.back());
            if (pCurrent->mParams.end() == iter)
            {
                // logging
                throw std::runtime_error("Key '" + keys.back() + "' not found");
            }

            try
            {
                return std::any_cast<T>(iter->second);
            }
            catch (const std::bad_any_cast &)
            {
                // logging
                throw std::runtime_error("Type mismatch for key '" + keys.back() + "'");
            }
        }
    }

    template<typename T>
    T GetParam(const std::string &key, const T &defaultValue) const
    {
        try
        {
            return GetParam<T>(key);
        }
        catch (...)
        {
            return defaultValue;
        }
    }

    bool HasParam(const std::string &key)
    {
        try
        {
            // 尝试获取参数, 但不关心值
            GetParam<std::any>(key);
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    void RemoveParam(const std::string &key)
    {
        size_t dot_pos = key.find('.');
        if (std::string::npos == dot_pos)
        {
            mParams.erase(key);
        }
        else
        {
            // 嵌套键, 需要递归处理
            std::vector<std::string> keys;
            size_t                   start = 0;
            while (dot_pos != std::string::npos)
            {
                keys.emplace_back(key.substr(start, dot_pos - start));
                start   = dot_pos + 1;
                dot_pos = key.find('.', start);
            }
            keys.emplace_back(key.substr(start));

            ParameterManager *pCurrent = this;
            for (size_t idx{0}; idx < keys.size() - 1; ++idx)
            {
                auto iter = pCurrent->mParams.find(keys[idx]);
                if (iter == pCurrent->mParams.end())
                {
                    return; // 中间键不存在，无需删除
                }

                try
                {
                    pCurrent = &std::any_cast<ParameterManager &>(iter->second);
                }
                catch (const std::bad_any_cast &)
                {
                    return; // 中间键不是字典类型，无法继续
                }
            }

            // 删除最后一级键
            pCurrent->mParams.erase(keys.back());
        }
    }

    void Clear()
    {
        mParams.clear();
    }

    std::vector<std::string> GetAllKeys() const
    {
        std::vector<std::string> keys;
        for (const auto &pair : mParams)
        {
            keys.emplace_back(pair.first);
        }
        return keys;
    }

private:
    std::unordered_map<std::string, std::any> mParams;

    // 递归设置嵌套参数的辅助函数
    template<typename T>
    void set_nested_param_impl(const std::string &key, T &&value, std::vector<std::string> &keys, size_t index)
    {
        if (keys.size() - 1 == index)
        {
            // 最后一级键, 设置值
            mParams[keys[index]] = std::forward<T>(value);
        }
        else
        {
            // 中间键, 需要保证是 'ParamManager' 类型
            auto iter = mParams.find(keys[index]);
            if (mParams.end() == iter)
            {
                // 不存在则创建新的嵌套管理器
                iter = mParams.emplace(keys[index], ParameterManager()).first;
            }

            // std::any 采用 std::any_cast 只支持抛出 exception, 所以采用 try-catch
            try
            {
                // 获取嵌套管理器并递归设置
                auto &nested = std::any_cast<ParameterManager &>(iter->second);
                nested.set_nested_param_impl(key, std::forward<T>(value), keys, index + 1);
            }
            catch (const std::bad_any_cast &ex)
            {
                // logging
                throw std::runtime_error("Key '" + keys[index] + "' is not a nested dictionary");
            }
        }
    }
};
