/**
 * @file 11_9_1_random_walk.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-07
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "11_9_1_my_vector.hpp"

#include <cstdlib> // rand(), srand() prototypes
#include <ctime>   // time() prototype
#include <fstream>
#include <iostream>
#include <string>


/**
 * @brief 编写C++程序, 将数据写入文件
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // show_time();
    std::cout << "==============================\n\n";

    /* --------------------------------- */
    std::string   filename = "./random_walk.txt";
    std::ofstream file_writer;
    file_writer.open(filename, std::ios::out);
    // file_writer.open(filename, std::ios::trunc);
    // file_writer.open(filename, std::ios::app);
    if (!file_writer.is_open())
    {
        std::cout << "open file is not successfully, please check." << filename << "\n";
        return 0;
    }

    std::srand(time(0)); /* seed random-number generator */

    double         direction;
    VECTOR::Vector step;
    VECTOR::Vector result(0.0, 0.0);

    unsigned long steps = 0;
    double        target;
    double        distance_step;

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

            file_writer << steps << ": (x, y) = "
                        << "(" << result.get_x_val() << ", " << result.get_y_val() << ")\n";
            ++steps;
        }
        std::cout << "After " << steps
                  << " steps, the subject "
                     "has the following location:\n";
        std::cout << result << std::endl;

        file_writer << "After " << steps
                    << " steps, the subject "
                       "has the following location:\n"
                    << "(x, y) = (" << result.get_x_val() << ", " << result.get_y_val() << ")";

        result.polar_mode();

        std::cout << " or\n" << result << std::endl;

        file_writer << "(m, a) = (" << result.get_mag_val() << ", " << result.get_ang_val() << ")";

        std::cout << "Average outward distance per step = " << result.get_mag_val() / steps << std::endl;

        file_writer << "Average outward distance per step = " << result.get_mag_val() / steps << std::endl;

        steps = 0;
        result.reset(0.0, 0.0);
        std::cout << "Enter target distance (q to quit): ";
    }
    std::cout << "==== Bye!\n";
    file_writer.close();

    std::cin.clear();
    while (std::cin.get() != '\n') continue;
    /* ------------------------------------- */

    return 0;
}
