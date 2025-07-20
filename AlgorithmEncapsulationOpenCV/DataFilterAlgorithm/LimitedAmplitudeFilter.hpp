/**
 * @file LimitedAmplitudeFilter.hpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 限幅滤波算法
 * @version 0.1
 * @date 2025-07-19
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#pragma once

/**
 * @brief 限幅滤波算法
 * 限幅数值可以根据实际情况进行调整
 * RunFilter API 实时输入当前采样值, 滤波后返回有效的实际值
 *
 * 优点: 能够有效克服因偶然因素引起的脉冲干扰
 * 缺点: 无法抑制那种周期性干扰,而且平滑度差
 *  
 */
template<typename DataType>
class LimitedAmplitudeFilter
{
public:
    DataType RunFilter(DataType value)
    {
        if (mPrevValue - value > mLimitedAmplitude || value - mPrevValue > mLimitedAmplitude)
        {
            // filter the exception data value via limited amplitude
            return mPrevValue;
        }
        else
        {
            mPrevValue = value;
            return value;
        }
    }

    void SetLimitedAmplitude(const DataType limitedAmplitude)
    {
        mLimitedAmplitude = limitedAmplitude;
    }

private:
    DataType mPrevValue;
    DataType mLimitedAmplitude;

public:
    LimitedAmplitudeFilter(const DataType initValue, const DataType limitedAmplitude)
        : mPrevValue{initValue}
        , mLimitedAmplitude{limitedAmplitude}
    {
    }

    LimitedAmplitudeFilter()  = delete;
    ~LimitedAmplitudeFilter() = default;
};
