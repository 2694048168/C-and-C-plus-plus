/**
 * @file MedianFilter.hpp
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
#include <vector>

/**
 * @brief 中位值滤波器
 * N 可以根据实际情况进行调整
 * 排序采用冒泡算法
 * 
 * 优点: 能够有效克服因偶然因素引起的波动干扰;
 *     对温度, 液位等变化缓慢的被测参数有良好的滤波效果;
 * 缺点: 对流量, 速度等快速变化的参数不宜;
 * 
 */
template<typename DataType>
class MedianFilter
{
public:
    DataType RunFilter()
    {
        std::sort(mData.begin(), mData.end());

        const auto SIZE = mData.size();
        if (0 == SIZE % 2)
        {
            return (mData[SIZE / 2] + mData[SIZE / 2 + 1]) / 2.0;
        }
        else
        {
            return mData[(SIZE - 1) / 2];
        }
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
    MedianFilter()  = default;
    ~MedianFilter() = default;
};

/* 
const int N = 11;

char filter(void)
{
    char value_buf[N];
    char i, j, temp;
    for (i = 0; i < N, ++i)
    {
        value_buf[i] = get_ad();
        delay();
    }

    for (j = 0; j < N - 1; ++j)
    {
        for (i = 0; i < N - j; ++i)
        {
            if (value_buf[i] > value_buf[i + j])
            {
                temp             = value_buf[i];
                value_buf[i]     = value_buf[i + 1];
                value_buf[i + 1] = temp;
            }
        }
    }

    return value_buf[(N - 1) / 2];
}
*/
