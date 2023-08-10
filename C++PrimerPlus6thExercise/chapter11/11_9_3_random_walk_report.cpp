/**
 * @file 11_9_3_random_walk_report.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_1_my_vector.hpp"

#include <algorithm>
#include <cstdlib> // rand(), srand() prototypes
#include <ctime>   // time() prototype
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

/**
 * @brief 编写C++程序, 将数据写入文件, 汇报 N 次测试的最大值/最小值/和平均值
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "==============================\n\n";

    /* --------------------------------- */
    std::string   filename = "./random_walk_report.txt";
    std::ofstream file_writer;
    file_writer.open(filename, std::ios::out);
    if (!file_writer.is_open())
    {
        std::cout << "open file is not successfully, please check." << filename << "\n";
        return -1;
    }

    std::srand(time(0)); /* seed random-number generator */

    double         direction;
    VECTOR::Vector step;
    VECTOR::Vector result(0.0, 0.0);

    unsigned long              steps = 0;
    std::vector<unsigned long> vec_steps;

    double target;
    double distance_step;

    std::cout << "Enter target distance (q to quit): ";
    while (std::cin >> target)
    {
        file_writer << "Target Distance: " << target;

        std::cout << "Enter step length: ";
        if (!(std::cin >> distance_step))
            break;

        file_writer << ", Step Size: " << distance_step << "\n";

        while (result.get_mag_val() < target)
        {
            direction = rand() % 360;
            step.reset(distance_step, direction, VECTOR::Vector::POL);
            result = result + step;

            ++steps;
        }

        result.polar_mode();

        vec_steps.push_back(steps);
        steps = 0;
        result.reset(0.0, 0.0);
        std::cout << "Enter target distance (q to quit): ";
    }
    std::cout << "==== Bye!\n";

    std::cin.clear();
    while (std::cin.get() != '\n') continue;

    file_writer << "The maximum Step : " << *std::max_element(vec_steps.cbegin(), vec_steps.cend()) << "\n";

    file_writer << "The minimum Step : " << *std::min_element(vec_steps.cbegin(), vec_steps.cend()) << "\n";

    file_writer << "The average Step : " << std::accumulate(vec_steps.cbegin(), vec_steps.cend(), 0) / vec_steps.size()
                << "\n";

    file_writer.close();
    /* ------------------------------------- */

    return 0;
}
