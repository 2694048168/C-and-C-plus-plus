/**
 * @file 6_11_7_alpha_classification.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-01
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cctype>
#include <iostream>
#include <string>

/**
 * @brief 编写C++程序, 统计用户输入的元音单词和辅音单词
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    unsigned int vowels     = 0;
    unsigned int consonants = 0;
    unsigned int other      = 0;

    std::cout << "Enter words (q to quit): ";
    std::string input;
    while (std::cin >> input)
    {
        if (input == "q")
            break;

        if (isalpha(input[0]))
        {
            switch (toupper(input[0]))
            {
            case 'A':;
            case 'E':;
            case 'I':;
            case 'O':;
            case 'U':
                ++vowels;
                break;

            default:
                ++consonants;
                break;
            }
        }
        else
        {
            ++other;
        }
    }

    std::cout << vowels << " words beginning with vowels.\n"
              << consonants << " words beginning with consonants.\n"
              << other << " words beginning with other letter." << std::endl;

    return 0;
}