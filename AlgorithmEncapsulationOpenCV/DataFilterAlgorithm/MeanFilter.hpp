/**
 * @file MeanFilter.hpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 中位值滤波器
 * @version 0.1
 * @date 2025-07-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

/**
 * @brief 算术平均滤波器
 * 
 * 优点: 适用对一般具有随机干扰的信号进行滤波;
 *     该信号特点是具有一个平均值, 信号在某一数值范围上下波动;
 * 缺点: 对于测量速度较慢或者要求数据计算较快的实时控制中不适用;
 * 
 */
template<typename DataType>
class MeanFilter
{
public:
    DataType RunFilter()
    {
        const auto SUM  = std::reduce(mData.begin(), mData.end(), 0.0);
        const auto SIZE = mData.size();
        return SUM / SIZE;
    }

    void SetDataVec(const std::vector<DataType> &data)
    {
        mData = data;
    }

    void AddData(const DataType &value)
    {
        mData.emplace_back(value);
    }

private:
    std::vector<DataType> mData;

public:
    MeanFilter()  = default;
    ~MeanFilter() = default;
};
