/**
 * @file 5_9_10_nesting_loop.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

/**
 * @brief 编写C++程序, 根据用户的输入, 嵌套循环打印行数星号 *
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Please enter the number of rows with an integer: ";
    unsigned int num_rows;
    std::cin >> num_rows;

    std::string info;
    // -----------------------------------------
    for (size_t idx_row = 0; idx_row < num_rows; ++idx_row)
    {
        std::string line_info(num_rows - idx_row - 1, '.');
        for (size_t idx_col = 0; idx_col < idx_row + 1; ++idx_col)
        {
            line_info += "*";
        }
        // std::cout << line_info << "\n";
        line_info += "\n";
        info += line_info;
    }
    // ----------- 改进方式, 减少循环嵌套 ------------------------
    // for (size_t idx_row = 0; idx_row < num_rows; ++idx_row)
    // {
    //     std::string line_info(num_rows - idx_row - 1, '.');
    //     // num_rows - idx_row - 1 + X = num_rows ---> X = idx_row + 1
    //     line_info += std::string(idx_row + 1, '*');
    //     line_info += "\n";
    //     info += line_info;
    // }
    // -----------------------------------------
    std::cout << info;

    return 0;
}