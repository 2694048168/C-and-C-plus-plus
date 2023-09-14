/**
 * @file 18_12_4_lambda_expression.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <vector>

const long SIZE = 390000L;

template<typename T>
void display_element(const std::vector<T> &list)
{
    for (const auto &elem : list)
    {
        std::cout << elem << " ";
    }
    std::cout << "\n";
}

/**
 * @brief 编写C++程序, 使用 Lambda 表达式
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<int> numbers(SIZE);

    std::srand(std::time(0));
    std::generate(numbers.begin(), numbers.end(), std::rand);
    std::cout << "Sample size = " << SIZE << '\n';

    // using lambdas
    int count3 = std::count_if(numbers.begin(), numbers.end(), [](int x) { return x % 3 == 0; });
    std::cout << "Count of numbers divisible by 3: " << count3 << '\n';

    int count13 = 0;
    std::for_each(numbers.begin(), numbers.end(), [&count13](int x) { count13 += x % 13 == 0; });
    std::cout << "Count of numbers divisible by 13: " << count13 << '\n';

    // using a single lambda
    count3 = count13 = 0;
    std::for_each(numbers.begin(), numbers.end(),
                  [&](int x)
                  {
                      count3 += x % 3 == 0;
                      count13 += x % 13 == 0;
                  });
    std::cout << "Count of numbers divisible by 3: " << count3 << '\n';
    std::cout << "Count of numbers divisible by 13: " << count13 << '\n';

    // =======================================================================
    std::vector<int> vec_list{1, 2, 3, 4, 5, 6, 7};

    auto print = [&vec_list]()
    {
        for (const auto &elem : vec_list)
        {
            std::cout << elem << " ";
        }
        std::cout << "\n";
    };

    print();
    std::cout << "-----------------\n";
    display_element(vec_list);

    return 0;
}