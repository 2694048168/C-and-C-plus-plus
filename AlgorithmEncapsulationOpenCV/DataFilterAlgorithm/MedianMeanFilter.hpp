/**
 * @file MedianMeanFilter.hpp
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
 * @brief 中位值平均滤波器
 * N 可以根据实际情况进行调整
 * 
 * 优点: 融合两种滤波优点, 对于偶然出现的脉冲性干扰, 可以消除其引起的采样偏差;
 *     对周期性干扰有良好的抑制作用, 平滑度高, 适用于高频震荡系统;
 * 缺点: 测量速度慢;
 * 
 */
template<typename DataType>
class MedianMeanFilter
{
public:
    DataType RunFilter()
    {
        std::sort(mData.begin(), mData.end());

        const auto SUM  = std::reduce(mData.begin(), mData.end(), 0.0);
        const auto SIZE = mData.size();

        return SUM / (SIZE - 2);
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
    MedianMeanFilter()  = default;
    ~MedianMeanFilter() = default;
};
