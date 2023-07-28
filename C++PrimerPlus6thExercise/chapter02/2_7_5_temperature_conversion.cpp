/**
 * @file 2_7_5_temperature_conversion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-25
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 摄氏温度转化为对应的华氏温度
 * 
 * @param celsius_degree 摄氏温度
 * @param fahrenheit_degree 华氏温度
 */
void temperature_conversion(const float celsius_degree, float &fahrenheit_degree)
{
    fahrenheit_degree = 1.8 * celsius_degree + 32.0;
}

float temperature_conversion(const float celsius_degree)
{
    // float fahrenheit = 1.8 * celsius_degree + 32.0;
    // return fahrenheit;

    return 1.8 * celsius_degree + 32.0;
}

/**
 * @brief 编写C++程序, main 函数调用用户自定义函数, 
 * 该函数以摄氏温度为参数, 并返回对应的华氏温度:
 * 华氏温度 = 1.8 * 摄氏温度 + 32.0 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "the temperature conversion(from Celsius into Fahrenheit):\n";
    std::cout << "Please enter a Celsius value: ";

    float celsius_value = 0.0f;
    std::cin >> celsius_value;

    // float fahrenheit_value = temperature_conversion(celsius_value);

    float fahrenheit_value = 0.0f;
    temperature_conversion(celsius_value, fahrenheit_value);

    std::cout << celsius_value << " degrees Celsius is " << fahrenheit_value << " degrees Fahrenheit." << std::endl;

    return 0;
}