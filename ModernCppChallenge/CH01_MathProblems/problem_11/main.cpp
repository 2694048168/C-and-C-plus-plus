/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-24
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <utility>
#include <vector>

/* Converting numerical values to Roman
Write a program that, given a number entered by the user, 
prints its Roman numeral equivalent. 
--------------------------------------- */

/* Solution
Roman numerals, as they are known today, 
use seven symbols: I = 1, V = 5, X = 10, L = 50, C = 100, D = 500, and M = 1000.
The system uses additions and subtractions in composing the numerical symbols. 
The symbols from 1 to 10 are I, II, III, IV, V, VI, VII, VIII, IX, and X.
Romans did not have a symbol for zero and used to write nulla to represent it.
In this system, the largest symbols are on the left, 
and the least significant are on the right. 
As an example, the Roman numeral for 1994 is MCMXCIV. 
-------------------------------------------------------- */
std::string to_roman(unsigned int value)
{
    std::vector<std::pair<unsigned int, const char *>> roman{
        {1000,  "M"},
        { 900, "CM"},
        { 500,  "D"},
        { 400, "CD"},
        { 100,  "C"},
        {  90, "XC"},
        {  50,  "L"},
        {  40, "XL"},
        {  10,  "X"},
        {   9, "IX"},
        {   5,  "V"},
        {   4, "IV"},
        {   1,  "I"}
    };

    std::string result;
    for (const auto &kvp : roman)
    {
        while (value >= kvp.first)
        {
            result += kvp.second;
            value -= kvp.first;
        }
    }

    return result;
}

// -----------------------------
int main(int argc, char **argv)
{
    for (int i = 1; i <= 100; ++i)
    {
        std::cout << i << "\t" << to_roman(i) << '\n';
    }
    std::cout << "==========================================\n";

    int number = 0;
    std::cout << "Pleas enter the number: ";
    std::cin >> number;

    std::cout << to_roman(number) << std::endl;

    return 0;
}
