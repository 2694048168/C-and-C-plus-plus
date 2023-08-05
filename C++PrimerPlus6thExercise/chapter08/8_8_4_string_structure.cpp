/**
 * @file 8_8_4_string_structure.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <string.h>

#include <cstring> // for strlen(), strcpy()
#include <iostream>

struct stringy
{
    char *str; // points to a string
    int   ct;  // length of string (not counting '\0')
};

// prototypes for set(), show(), and show() go here
void set(stringy &str, char *arr)
{
    unsigned len = strlen(arr);

    str.ct = len;
    // memory allocation of "len" number of chars
    str.str = new char[len + 1];

    // strcpy(str.str, arr);
    strcpy_s(str.str, len + 1, arr);
}

void show(const char *str, unsigned int num = 1)
{
    std::cout << str << std::endl;
}

void show(const stringy &str, unsigned int num = 1)
{
    std::cout << str.str << std::endl;
}

/**
 * @brief 编写C++程序, 完成程式的补充
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    char testing[] = "Reality isn't what it used to be.";

    stringy bean;
    set(bean, testing);

    show(bean);
    show(bean, 2);

    testing[0] = 'D';
    testing[1] = 'u';

    show(testing);
    show(testing, 3);

    show("Done!");

    return 0;
}