/**
 * @file 6_11_6_track_contributions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-31
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>
#include <vector>

struct Contribution
{
    std::string name;
    double      money;
};

/**
 * @brief 编写C++程序, 记录捐赠人的相关信息, 并按照特定要求进行显示结果
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<Contribution> contribution_vec;

    auto print_help = []()
    {
        std::cout << "=====================================================\n";
        std::cout << "Please selection some operator as following:\n";
        std::cout << "i. input information to store\n";
        std::cout << "p. print information for contribution\n";
        std::cout << "h. print the help menu information\n";
        std::cout << "q. quit to exit!\n";
        std::cout << "=====================================================\n";
    };
    print_help();

    auto print_info = [&contribution_vec]()
    {
        // > 10000 ---> Grand Patrons; <= 10000 ---> Patrons;
        std::vector<unsigned int> num_GrandPatrons;
        std::vector<unsigned int> num_Patrons;

        for (size_t idx = 0; idx < contribution_vec.size(); ++idx)
        {
            if (contribution_vec[idx].money > 10000)
            {
                num_GrandPatrons.push_back(idx);
            }
            else
            {
                num_Patrons.push_back(idx);
            }
        }

        std::cout << "========= Grand Patrons =========\n";
        if (num_GrandPatrons.empty())
        {
            std::cout << "None\n";
        }
        else
        {
            for (const auto elem : num_GrandPatrons)
            {
                std::cout << "Name: " << contribution_vec[elem].name;
                std::cout << " Money: " << contribution_vec[elem].money << "\n";
            }
        }

        std::cout << "=========       Patrons =========\n";
        if (num_Patrons.empty())
        {
            std::cout << "None\n";
        }
        else
        {
            for (const auto elem : num_Patrons)
            {
                std::cout << "Name: " << contribution_vec[elem].name;
                std::cout << " Money: " << contribution_vec[elem].money << "\n";
            }
        }
    };

    char        input = 0;
    std::string name  = "wei li";
    double      money = 0;
    std::cin >> input;
    std::cin.ignore();
    while (input != 'q')
    {
        if (input == 'i')
        {
            std::cout << "Please enter the name of contribution: ";
            std::getline(std::cin, name);
            std::cout << "Please enter the money of contribution: ";
            std::cin >> money;
            contribution_vec.push_back({name, money});

            std::cout << "Please selection some operator as following:";
            std::cin >> input;
            std::cin.ignore();
        }
        else if (input == 'p')
        {
            print_info();

            std::cout << "Please selection some operator as following:";
            std::cin >> input;
            std::cin.ignore();
        }
        else if (input == 'h')
        {
            print_help();
            std::cin >> input;
            std::cin.ignore();
        }
        else
        {
            std::cout << "This function is not implement.\n";
            print_help();
            std::cin >> input;
            std::cin.ignore();
        }
    }
    std::cout << "Program exit successfully." << std::endl;

    return 0;
}