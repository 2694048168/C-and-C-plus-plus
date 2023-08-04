/**
 * @file 7_13_8_array_function.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-03
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <array>
#include <iostream>
#include <string>

const int Seasons = 4;

// const std::array<std::string, Seasons> Snames = {"Spring", "Summer", "Fall", "Winter"};
const char* Snames[Seasons] = {"Spring", "Summer", "Fall", "Winter"};

void fill(std::array<double, Seasons> *pa);
void show(std::array<double, Seasons> da);

int main(int argc, const char **argv)
{
    std::array<double, 4> expenses;

    fill(&expenses);
    show(expenses);

    return 0;
}

void fill(std::array<double, Seasons> *pa)
{
    for (int i = 0; i < Seasons; i++)
    {
        std::cout << "Enter " << Snames[i] << " expenses: ";
        std::cin >> (*pa)[i];
    }
}

void show(std::array<double, Seasons> da)
{
    double total = 0.0;
    std::cout << "\nEXPENSES\n";
    for (int i = 0; i < Seasons; i++)
    {
        std::cout << Snames[i] << ": $" << da[i] << '\n';
        total += da[i];
    }
    std::cout << "Total: $" << total << '\n';
}