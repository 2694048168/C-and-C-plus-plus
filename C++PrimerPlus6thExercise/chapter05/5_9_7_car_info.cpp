/**
 * @file 5_9_7_car_info.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief 编写C++程序, 记录 car 的相关信息, 利用标准 IO 操作.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    struct Car
    {
        std::string  producer;
        unsigned int year;
    };

    std::cout << "How many cars do you wish to catalog? ";
    unsigned count = 0;
    std::cin >> count;
    std::cin.ignore();

    // -------------------------------
    std::vector<Car *> vec_cat_pointer(count);
    for (size_t i = 0; i < count; ++i)
    {
        vec_cat_pointer[i] = new Car();
        // TODO 是否存在 memory leaky, 是否需要进行处理 delete?

        std::cout << "Car #" << (i + 1) << ":\n";
        std::cout << "Please enter the make: ";
        std::getline(std::cin, vec_cat_pointer[i]->producer);
        std::cout << "Please enter the year made: ";
        std::cin >> vec_cat_pointer[i]->year;
        std::cin.ignore();
    }

    std::cout << "\nHere is your collection:\n";
    for (const auto car : vec_cat_pointer)
    {
        std::cout << car->year << " " << car->producer << "\n";
    }
    std::cout << "----------------------------" << std::endl;

    // -------------------------------
    // std::vector<Car> vec_car(count);
    // for (size_t i = 0; i < count; ++i)
    // {
    //     std::cout << "Car #" << (i + 1) << ":\n";
    //     std::cout << "Please enter the make: ";
    //     std::getline(std::cin, vec_car[i].producer);
    //     std::cout << "Please enter the year made: ";
    //     std::cin >> vec_car[i].year;
    //     std::cin.ignore();
    // }

    // std::cout << "\nHere is your collection:\n";
    // for (const auto car : vec_car)
    // {
    //     std::cout << car.year << " " << car.producer << "\n";
    // }
    // std::cout << "----------------------------" << std::endl;

    return 0;
}