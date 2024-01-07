/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Generating all the permutations of a string
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <iostream>
#include <string>

/**
 * @brief Generating all the permutations of a string
 * 
 * Write a function that, prints on the console all the possible permutations 
 * of a given string. You should provide two versions of this function: 
 * one that uses recursion, and one that does not.
 * 
 * You can solve this problem by taking advantage of some general-purpose algorithms 
 * from the standard library. The simplest of the two required versions is 
 * the non-recursive one, at least when you use std::next_permutation(). 
 * This function transforms the input range (that is required to be sorted) into
 * the next permutation from the set of all possible permutations, 
 * ordered lexicographically with operator< or the specified comparison function object.
 * If such a permutation exists then it returns true, otherwise, 
 * it transforms the range into the first permutation and returns false. 
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
void print_permutations(std::string str)
{
    std::sort(std::begin(str), std::end(str));

    do
    {
        std::cout << str << std::endl;
    }
    while (std::next_permutation(std::begin(str), std::end(str)));
}

void next_permutation(std::string str, std::string perm)
{
    if (str.empty())
        std::cout << perm << std::endl;
    else
    {
        for (size_t i = 0; i < str.size(); ++i)
        {
            next_permutation(str.substr(1), perm + str[0]);

            std::rotate(std::begin(str), std::begin(str) + 1, std::end(str));
        }
    }
}

void print_permutations_recursive(std::string str)
{
    next_permutation(str, "");
}

// ------------------------------
int main(int argc, char **argv)
{
    std::cout << "non-recursive version" << std::endl;
    print_permutations("main");

    std::cout << "recursive version" << std::endl;
    print_permutations_recursive("main");

    return 0;
}
