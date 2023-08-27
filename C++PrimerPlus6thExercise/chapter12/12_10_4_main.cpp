/**
 * @file 12_10_4_main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "12_10_4_stack.hpp"

#include <iostream>

/**
 * @brief 编写C++程序, TODO 使用动态内存分配
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Stack         st; // create an empty stack
    char          ch;
    unsigned long po;
    std::cout << "Please enter A to add a purchase order,\n"
              << "P to process a PO, or Q to quit.\n";
    while (std::cin >> ch && toupper(ch) != 'Q')
    {
        while (std::cin.get() != '\n') continue;
        if (!isalpha(ch))
        {
            std::cout << '\a';
            continue;
        }
        switch (ch)
        {
        case 'A':
        case 'a':
            std::cout << "Enter a PO number to add: ";
            std::cin >> po;
            if (st.is_full())
                std::cout << "stack already full\n";
            else
                st.push(po);
            break;
        case 'P':
        case 'p':
            if (st.is_empty())
                std::cout << "stack already empty\n";
            else
            {
                st.pop(po);
                std::cout << "PO #" << po << " popped\n";
            }
            break;
        }
        std::cout << "Please enter A to add a purchase order,\n"
                  << "P to process a PO, or Q to quit.\n";
    }
    std::cout << "Bye\n";

    return 0;
}