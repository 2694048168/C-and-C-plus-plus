/**
 * @file 4_13_1_show_information.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-07-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>
#include <vector>

/**
 * @brief 编写C++程序, 输出指定用户信息, 包括成绩的下调变化
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "What is your first name? ";
    std::string first_name = "Li";
    // std::cin >> first_name;
    // 可以接受带有空格的字符串,getline function
    // It reads input through the newline character marking the end of the line,
    //  but it doesn’t save the newline character. 
    // Instead, it replaces it with a null character when storing the string.
    std::getline(std::cin, first_name);

    std::cout << "What is your last name? ";
    std::string last_name = "Wei";
    std::getline(std::cin, last_name);

    std::cout << "What letter grade(A, B, or C) do you deserve? ";
    char grade = 'A';
    std::cin >> grade;

    // 利用 array 类型实现成绩等级下调,
    // const char grade_levels[] = {'A', 'B', 'C', 'D'};

    // const size_t size = sizeof(grade_levels) / sizeof(grade_levels[0]);
    // for (size_t i = 0; i < size; ++i)
    // {
    //     if (grade == grade_levels[i])
    //     {
    //         grade = grade_levels[i + 1];
    //         break;
    //     }
    //     else
    //     {
    //         continue;
    //     }
    // }
    // ----------------------------------------

    // 利用 vector 类型实现成绩等级下调,
    const std::vector<char> levels_grade = {'A', 'B', 'C', 'D'};
    for (size_t i = 0; i < levels_grade.size(); ++i)
    {
        if (grade == levels_grade[i])
        {
            grade = levels_grade[i + 1];
            break;
        }
    }

    std::cout << "What is your age? ";
    unsigned int age = 0;
    std::cin >> age;

    std::cout << "Name: " << last_name << ", " << first_name << "\nGrade: " << grade << "\nAge: " << age << std::endl;

    return 0;
}