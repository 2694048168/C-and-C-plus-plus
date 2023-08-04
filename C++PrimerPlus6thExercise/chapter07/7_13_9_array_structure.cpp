/**
 * @file 7_13_9_array_structure.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string.>
#include <string>

// const unsigned int SLEN = 30;
struct student
{
    // char fullname[SLEN];
    // char hobby[SLEN];
    std::string fullname;
    std::string hobby;
    int         oop_level;
};

int getinfo(student pa[], int n);

void display1(student st);
void display2(const student *ps);
void display3(const student pa[], int n);

/**
 * @brief 编写C++程序, 利用函数处理数组和结构体
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    std::cout << "Enter class size : ";
    int class_size;
    std::cin >> class_size;
    while (std::cin.get() != '\n') continue;

    student *ptr_stu = new student[class_size];

    int entered = getinfo(ptr_stu, class_size);

    for (int i = 0; i < entered; i++)
    {
        display1(ptr_stu[i]);
        display2(&ptr_stu[i]);
    }

    display3(ptr_stu, entered);

    delete[] ptr_stu;

    return 0;
}

int getinfo(student pa[], int n)
{
    std::string fullname;
    std::string hobby;

    int oop_level;
    int idx;
    for (idx = 0; idx < n; ++idx)
    {
        std::cout << "Enter the fullname #" << (idx + 1) << ": ";
        std::getline(std::cin, fullname);

        std::cout << "Enter the hobby #" << (idx + 1) << ": ";
        std::getline(std::cin, hobby);

        std::cout << "Enter the OOP level #" << (idx + 1) << ": ";
        std::cin >> oop_level;
        std::cin.ignore();

        pa[idx] = {fullname, hobby, oop_level};
    }

    return idx;
}

void display1(student st)
{
    std::cout << "=======================================\n";
    std::cout << "The fullname of student: " << st.fullname;
    std::cout << "\nThe hobby of student: " << st.hobby;
    std::cout << "\nThe OOP level of student: " << st.oop_level << std::endl;
}

void display2(const student *ps)
{
    std::cout << "=======================================\n";
    std::cout << "The fullname of student: " << ps->fullname;
    std::cout << "\nThe hobby of student: " << ps->hobby;
    std::cout << "\nThe OOP level of student: " << ps->oop_level << std::endl;
}

// n = std::min(entered, class_size)
void display3(const student pa[], int n)
{
    std::cout << "=======================================\n";

    // as array,
    for (size_t i = 0; i < n; ++i)
    {
        std::cout << "\nThe fullname of student: " << pa[i].fullname;
        std::cout << "\nThe hobby of student: " << pa[i].hobby;
        std::cout << "\nThe OOP level of student: " << pa[i].oop_level << std::endl;
    }

    // TODO as pointer, ++address, 需要考虑结构体内存布局以及访问
    std::cout << "=======================================\n";
}