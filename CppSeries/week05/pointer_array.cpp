/**
 * @file pointer_array.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief pointer and array
 * @attention
 *
 */

#include <iostream>

// struct like class in C++, not such as 'typedef' in C
struct Student
{
    char name[4];
    int born;
    bool male;
};

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    std::cout << "sizeof(Student) = " << sizeof(Student) << std::endl;
    // Part One
    Student students[128];
    Student *p0 = &students[0];
    Student *p1 = &students[1];
    Student *p2 = &students[2];
    Student *p3 = &students[3];

    printf("p0 = %p\n", p0);
    printf("p1 = %p\n", p1);
    printf("p2 = %p\n", p2);
    printf("p3 = %p\n", p3);
    std::cout << "-------------------------------" << std::endl;

    // the same behavior
    students[1].born = 2000;
    p1->born = 2000;

    // Part Two
    printf("&students = %p\n", &students);
    printf("students = %p\n", students);
    printf("&students[0] = %p\n", &students[0]);
    std::cout << "-------------------------------" << std::endl;

    // array-name as address of arry, so as the pointer?
    Student *p = students;
    p[0].born = 2000;
    p[1].born = 2001;
    p[2].born = 2002;

    printf("students[0].born = %d\n", students[0].born);
    printf("students[1].born = %d\n", students[1].born);
    printf("students[2].born = %d\n", students[2].born);
    std::cout << "-------------------------------" << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ pointer_array.cpp
 * $ clang++ pointer_array.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */