/**
 * @file 6_11_9_track_contributions.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Contribution
{
    std::string name;
    double      money;
};

/**
 * @brief 编写C++程序, 从文件中读取捐赠人的相关信息, 并按照特定要求进行显示结果
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
        std::cout << "r. read information from file\n";
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

    char input = 0;
    std::cin >> input;
    std::cin.ignore();
    while (input != 'q')
    {
        if (input == 'r')
        {
            // reading info from file, 'Patrons.txt'
            const char *filename = "./Patrons.txt";

            std::ifstream file_reader;
            file_reader.open(filename, std::ios::in);
            if (!file_reader.is_open())
            {
                std::cout << "read file is not successfully, please check." << filename << "\n";
                return -1;
            }

            std::string  buffer;
            unsigned int idx_row = 0;
            std::string  name;
            double       money;
            while (!file_reader.eof())
            {
                std::getline(file_reader, buffer);
                if (idx_row == 0)
                {
                    std::cout << "The total number of Patrons: " << buffer << "\n";
                    ++idx_row;
                }
                else if (idx_row % 2 == 1)
                {
                    name = buffer;
                    ++idx_row;
                }
                else /* idx_row % 2 == 0 */
                {
                    // Since C++11, header <string>
                    // std::stof() - convert string to float
                    // std::stod() - convert string to double
                    // std::stold() - convert string to long double.
                    // money = (double)buffer;
                    money = std::stod(buffer);
                    ++idx_row;

                    // 此时获取完整一组数据, 进行存储和记录,
                    contribution_vec.push_back({name, money});
                }
            }

            file_reader.close();
            /* ------------------------------------- */
            std::cout << "Reading information from file successfully\n";

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