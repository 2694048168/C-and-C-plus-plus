/**
 * @file firstclass.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the first Class in C++
 * @attention
 *
 */

#include <iostream>
#include <cstring>
#include <string>

class Student
{
public:
// private:
    char m_name[4] = "Li";
    int m_born = 1999;
    bool m_male = false;

public:
    void set_name(const char *name)
    {
        strncpy(m_name, name, sizeof(m_name));
    }
    std::string get_name()
    {
        return m_name;
    }

    void set_born(int born)
    {
        m_born = born;
    }
    int get_born()
    {
        return m_born;
    }

    void set_gender(int gender)
    {
        m_male = gender;
    }
    int get_gender()
    {
        return m_male;
    }

    void print_info()
    {
        std::cout << "Name: " << m_name << std::endl;
        std::cout << "Born in " << m_born << std::endl;
        std::cout << "Gender: " << (m_male ? "Male" : "Female") << std::endl;
    }
};

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    Student li;

    li.set_name("Li");
    li.set_born(2000);
    li.set_gender(true);
    li.print_info();

    li.m_born = 2001; // it can also be manipulated directly
    std::cout << "It's name is " << li.m_name << std::endl;

    return 0;
}


/** Build(compile and link) commands via command-line.
 *
 * $ clang++ firstclass.cpp
 * $ clang++ firstclass.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */