/**
 * @file 3_7_7_driving_conversion.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

float driving_conversion(const float European_style)
{
    // 1. 假设为 100 公里, 则总的耗油量为 European_style 升,
    // 2. 则可以计算出对应的车程数,
    float driving_distance = 62.14f;

    // 3. 可以计算出对应的耗油量为多少加仑,
    const float scale_gasoline = 1 / 3.875f;

    float gasoline_gallon = European_style * scale_gasoline;

    // 4. 则可以计算对应的美式风格,
    float US_style = driving_distance / gasoline_gallon;

    return US_style;
}

/**
 * @brief 编写C++程序, 要求用户按欧洲风格输入汽车的耗油量(每100公里消耗的汽油量(升)),
 * 然后将其转换为美国风格的耗油量-—每加仑多少英里.
 * 注意，除了使用不同的单位计量外: 美国方法(距离/燃料）与欧洲方法（燃料/距离）相反.
 *  100公里等于62.14英里, 1加伦仑等于3.875升
 *  因此, 19mpg大约合12.41/100km，127mpg大约合8.71/100km.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter the automobile gasoline consumption(European style): ";
    float European_style = 0.f;
    std::cin >> European_style;

    float US_style = driving_conversion(European_style);

    std::cout << "The automobile gasoline consumption(US style) " << US_style << std::endl;

    return 0;
}