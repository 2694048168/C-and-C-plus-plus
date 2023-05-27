/**
 * @file pair_stl.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-05-26
 * 
 * @copyright Copyright (c) 2023
 * 
 * ---------------------------------------------
 * the <utility> header Syntax: 
 * std::pair <data_type1, data_type2> Pair_name;
 * std::pair <data_type1, data_type2> Pair_name(value1, value2);
 * Pair_name = std::make_pair(value1, value2);
 * ---------------------------------------------
 * swap() and tie() memory function for std::pair
 * operator overloading for "operators(=, ==, !=, >=, <=)"
 * 
 * clang++ pair_stl.cpp -std=c++17
 * g++ pair_stl.cpp -std=c++17
 * 
 */

#include <iostream>
#include <string>
#include <utility>

/**
 * @brief the std::pair<> container in C++ STL, such for tuple in Python.
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    // ------------------------------
    std::pair<int, char> pair_1;
    pair_1.first  = 100;
    pair_1.second = 'W';

    std::cout << "the first of pair: " << pair_1.first << std::endl;
    std::cout << "the first of pair: " << pair_1.second << std::endl;

    std::pair<std::string, float> pair_2("WeiLi", 3.1415);
    std::cout << "\nthe value of pair: (" << pair_2.first << ", " << pair_2.second << ")\n";

    std::pair<double, std::string> pair_3;
    pair_3 = std::make_pair(1.23, "Hello");
    std::cout << "\nthe value of pair: (" << pair_3.first << ", " << pair_3.second << ")\n";

    // ------------------------------
    std::pair<int, char> pair1 = std::make_pair(65, 'A');
    std::pair<int, char> pair2 = std::make_pair(66, 'B');

    std::cout << "\nbefore swap\n";
    std::cout << "pair1: (" << pair1.first << ", " << pair1.second << ")\n";
    std::cout << "pair2: (" << pair2.first << ", " << pair2.second << ")\n";

    pair1.swap(pair2);

    std::cout << "\nafter swap\n";
    std::cout << "pair1: (" << pair1.first << ", " << pair1.second << ")\n";
    std::cout << "pair2: (" << pair2.first << ", " << pair2.second << ")\n";

    // ------------------------------
    std::pair<int, int> pair_4 = { 1, 2 };

    int a, b;
    std::tie(a, b) = pair_4;
    std::cout << "\n" << a << " " << b << "\n";
  
    std::pair<int, int> pair_5 = { 3, 4 };
    std::tie(a, std::ignore) = pair_5;
    // prints old value of b, 
    std::cout << a << " " << b << "\n";
  
    // Illustrating pair of pairs, tie() just for two element!
    std::pair<int, std::pair<int, char> > pair_6 = { 3, { 4, 'a' } };
    std::cout << "pair: (" << pair_6.first << ", " << pair_6.second.first << ")\n";

    return 0;
}