/**
 * @file 3_7_6_driving_distance.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写C++程序,要求用户输入车程数量和耗油量, 计算单位车程所耗油量.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter your driving distance: ";
    float distance_driving = 0.f;
    std::cin >> distance_driving;

    std::cout << "Please enter your driving gasoline: ";
    float gasoline_driving = 0.f;
    std::cin >> gasoline_driving;

    // TODO 考虑单位的一致性, 每一英里/耗油量加仑; 100公里/耗油量多少升
    std::cout << "The average gasoline for per distance is " << distance_driving / gasoline_driving << std::endl;

    return 0;
}