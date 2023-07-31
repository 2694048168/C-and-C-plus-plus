/**
 * @file 6_11_4_programmers_info.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <vector>

// Benevolent Order of Programmers name structure
const unsigned int strsize = 24;

struct bop
{
    char fullname[strsize]; // real name
    char title[strsize];    // job title
    char bop_name[strsize]; // secret BOP name
    int  preference;        // 0 = fullname, 1 = title, 2 = bop_name
};

void show_name(const std::vector<bop> &bop_vec);
void show_title(const std::vector<bop> &bop_vec);
void show_bop_name(const std::vector<bop> &bop_vec);
void show_preference(const std::vector<bop> &bop_vec);

/**
 * @brief 编写C++程序, 利用结构体完成程序员信息的采集和按需要进行展示
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<bop> bop_vec;
    bop_vec.push_back({"wei li", "reading", "CPlusPlus", 0});
    bop_vec.push_back({"san zhu", "paper", "CSharp", 1});
    bop_vec.push_back({"wen liu", "research", "Python", 2});

    // auto func = []() -> return_type { };
    auto print_info = []()
    {
        std::cout << "=====================================================\n";
        std::cout << "Please enter one of the following choices\n";
        std::cout << "a. display by name       \tb. display by title\n";
        std::cout << "c. display by bop_name   \td. display by preference\n";
        std::cout << "q. quit to exit!\n";
        std::cout << "=====================================================\n";
    };

    print_info();

    char ch;
    while (std::cin >> ch)
    {
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
            case 'a':
                std::cout << "Show the programmer information via fullname:\n";
                show_name(bop_vec);
                break;
            case 'b':
                std::cout << "Show the programmer information via title:\n";
                show_title(bop_vec);
                break;
            case 'c':
                std::cout << "Show the programmer information via bop_name:\n";
                show_bop_name(bop_vec);
                break;
            case 'd':
                std::cout << "Show the programmer information via preference:\n";
                show_preference(bop_vec);
                break;
            default:
                std::cout << "This function is not implement\n";
                break;
            }
        }
    }

    return 0;
}

void show_name(const std::vector<bop> &bop_vec)
{
    for (const auto elem : bop_vec)
    {
        std::cout << elem.fullname << "\n";
    }
    std::cout << "------------------------" << std::endl;
}

void show_title(const std::vector<bop> &bop_vec)
{
    for (const auto elem : bop_vec)
    {
        std::cout << elem.title << "\n";
    }
    std::cout << "------------------------" << std::endl;
}

void show_bop_name(const std::vector<bop> &bop_vec)
{
    for (const auto elem : bop_vec)
    {
        std::cout << elem.bop_name << "\n";
    }
    std::cout << "------------------------" << std::endl;
}

void show_preference(const std::vector<bop> &bop_vec)
{
    for (const auto elem : bop_vec)
    {
        if (elem.preference == 0)
        {
            std::cout << elem.fullname << "\n";
        }
        else if (elem.preference == 1)
        {
            std::cout << elem.title << "\n";
        }
        else
        {
            std::cout << elem.bop_name << "\n";
        }
    }
    std::cout << "------------------------" << std::endl;
}