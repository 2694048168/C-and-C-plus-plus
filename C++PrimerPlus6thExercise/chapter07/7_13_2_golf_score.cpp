/**
 * @file 7_13_2_golf_score.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-02
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <numeric>
#include <vector>

void set_golf_score(std::vector<float> &golf_vec);
void show_golf_score(const std::vector<float> &golf_vec);
void compute_golf_score(const std::vector<float> &golf_vec);

/**
 * @brief 编写C++程序, 要用用户输入高尔夫成绩, 进行显示和计算其均值
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::vector<float> golf_vec;

    set_golf_score(golf_vec);
    show_golf_score(golf_vec);
    compute_golf_score(golf_vec);

    return 0;
}

void set_golf_score(std::vector<float> &golf_vec)
{
    std::cout << "Please enter the scores of Golf:\n";
    float score = 0.f;
    std::cin >> score;
    // score == 0 to exit loop,
    // TODO 更好的退出条件或者提示
    while (score)
    {
        golf_vec.push_back(score);
        std::cin >> score;
    }
}

void show_golf_score(const std::vector<float> &golf_vec)
{
    for (const auto score : golf_vec)
    {
        std::cout << score << " ";
    }
    std::cout << std::endl;
}

void compute_golf_score(const std::vector<float> &golf_vec)
{
    // C++17 version with higher numerical stability
    float average = std::reduce(golf_vec.begin(), golf_vec.end()) / golf_vec.size();

    std::cout << "\nThe average score: " << average << std::endl;
}