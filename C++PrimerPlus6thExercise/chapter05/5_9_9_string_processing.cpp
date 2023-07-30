/**
 * @file 5_9_9_string_processing.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-30
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>
#include <vector>

template<typename T>
void print_vec(const std::vector<T> &vec)
{
    std::cout << "----------------\n";
    for (const auto elem : vec)
    {
        std::cout << elem << " ";
    }
    std::cout << "\n----------------\n";
}

/**
 * @brief 编写C++程序, 利用 char 数组对用户输入的单词进行统计, done 作为统计结束标志
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    unsigned total_words = 0;

    std::vector<std::string> word_vec;

    std::cout << "Enter words (to stop, type the word done):\n";
    // special character, called a sentinel character,
    // to act as a stop sign 'done'

    std::string word;
    std::cin >> word;
    while (word != "done")
    {
        word_vec.push_back(word);
        ++total_words;
        std::cin >> word;
    }

    std::cout << "You entered a total of " << total_words << " words.\n";
    std::cout << "\nYou entered valid word:\n";
    print_vec(word_vec);

    return 0;
}