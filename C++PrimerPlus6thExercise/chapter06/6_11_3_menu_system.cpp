/**
 * @file 6_11_3_menu_system.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iomanip>
#include <iostream>

void print_menu()
{
    std::cout << "=============================================\n";
    std::cout << "Please enter one of the following choices for\n";
    // TODO 如何利用 C++ IO 格式控制, 保证输出打印更加美观?
    std::cout << "c) carnivore    \tp) pianist\n";
    std::cout << "t) tree         \tg) game\n";
    std::cout << "Please enter q or Q to exit!\n";
    std::cout << "=============================================\n";
}

/**
 * @brief 编写C++程序, 根据用户的选择进行菜单的不同功能,
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    print_menu();

    char ch;
    while (std::cin >> ch)
    {
        // print_menu();
        if ((ch == 'q') || (ch == 'Q'))
        {
            // q or Q, exit the program.
            std::cout << "Program exit successfully." << std::endl;
            break;
        }
        else
        {
            switch (ch)
            {
            case 'c':
            case 'C':
                std::cout << "This is a function for carnivore.\n";
                break;
            case 'p':
            case 'P':
                std::cout << "This is a function for pianist.\n";
                break;
            case 't':
            case 'T':
                std::cout << "This is a function for tree.\n";
                break;
            case 'g':
            case 'G':
                std::cout << "This is a function for game.\n";
                break;
            default:
                std::cout << "This function is not implement\n";
                break;
            }
        }
    }

    return 0;
}