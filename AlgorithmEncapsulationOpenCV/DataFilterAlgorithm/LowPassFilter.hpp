/**
 * @file LowPassFilter.hpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 低通滤波器
 * @version 0.1
 * @date 2025-07-20
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

/**
 * @brief 低通滤波器
 * 
 */
template<typename DataType>
class LowPassFilter
{
public:
    DataType RunFilter(const DataType value)
    {
        return mAmplitude * value + (1 - mAmplitude) * mPrevValue;
    }

    void SetAmplitude(const double amplitude)
    {
        mAmplitude = amplitude;
    }

private:
    double   mAmplitude; // 0.0 ~ 1.0
    DataType mPrevValue;

public:
    LowPassFilter(const double amplitude, const DataType initValue)
        : mAmplitude{amplitude}
        , mPrevValue{initValue}
    {
    }

    LowPassFilter(const DataType initValue)
        : mPrevValue{initValue}
    {
    }

    LowPassFilter()  = default;
    ~LowPassFilter() = default;
};
