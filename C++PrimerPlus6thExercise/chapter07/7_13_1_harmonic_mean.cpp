/**
 * @file 7_13_1_harmonic_mean.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

inline float harmonic_mean(const float &num1, const float &num2);
inline bool  harmonic_mean(const float &num1, const float &num2, float &mean);

/**
 * @brief 编写C++程序, 用户输入两个数字, 计算其调和平均数
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter tow number to compute harmonic mean\n";
    std::cout << "================================================\n";

    std::cout << "The number is: ";
    float num1;
    float num2;
    float num;
    std::cin >> num;
    unsigned long long int idx = 1;
    while (num != 0)
    {
        if (idx % 2 == 0)
        {
            num2 = num;
            ++idx;

            // ----------------------------------------
            // float mean = harmonic_mean(num1, num2);
            // std::cout << "The harmonic mean for two number is: " << mean << std::endl;
            // ----------------------------------------
            float mean = 0.f;
            bool  flag = harmonic_mean(num1, num2, mean);
            if (flag)
            {
                std::cout << "The harmonic mean for two number is: " << mean << std::endl;
            }
            else
            {
                std::cout << "The compute of harmonic mean is not successfully.\n";
            }
            // ----------------------------------------

            std::cout << "The number is: ";
            std::cin >> num;
        }
        else
        {
            num1 = num;
            ++idx;

            std::cout << "The number is: ";
            std::cin >> num;
        }
    }

    return 0;
}

inline float harmonic_mean(const float &num1, const float &num2)
{
    // TODO 保证程序的健壮性, 应该对传入的参数进行检查, 是否同时为零等情况
    float mean = 2.0 * num1 * num2 / (num1 + num2);
    return mean;
}

inline bool harmonic_mean(const float &num1, const float &num2, float &mean)
{
    // 这一步的参数检查是否有必要?
    // 针对本程序好像没有意义, 因为传入参数必然满足需要?
    if (num1 == 0 && num2 == 0)
    {
        return false;
    }

    mean = 2.0 * num1 * num2 / (num1 + num2);
    return true;
}