/**
 * @file 4_13_7_pizza_analysis_service.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-28
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

struct PizzaService
{
    std::string name_pizza;
    float       diameter_pizza;
    float       weight_pizza;

    void print_info()
    {
        std::cout << "---------------------------------------\n";
        std::cout << "the name of pizza: " << name_pizza << "\n";
        std::cout << "the diameter of pizza: " << diameter_pizza << "\n";
        std::cout << "the weight of pizza: " << weight_pizza << "\n";
    }
};

/**
 * @brief 编写C++程序, 存储披萨该物体, 并对此进行输入和输出
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // variable on stack for memory layout,
    PizzaService su;

    std::cout << "Please enter the name of 'su' pizza: ";
    std::getline(std::cin, su.name_pizza);

    std::cout << "Please enter the diameter(cm) of 'su' pizza: ";
    std::cin >> su.diameter_pizza;

    std::cout << "Please enter the weight(g) of 'su' pizza: ";
    std::cin >> su.weight_pizza;

    su.print_info();

    return 0;
}