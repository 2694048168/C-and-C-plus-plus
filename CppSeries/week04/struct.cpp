/**
 * @file struct.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief struct data type
 * @attention
 *
 */

#include <iostream>
#include <cstring>

struct Student
{
    char name[4];
    int born;
    bool male;
};

typedef struct _Student1
{
    int id;
    bool male;
    char label;
    float weight;
} Student1;

typedef struct _Student2
{
    int id;
    bool male;
    float weight;
    char label;
} Student2;

typedef unsigned char vec3b[3];
typedef struct _rgb_struct
{ // name _rgb_struct can be omit
    unsigned char r;
    unsigned char g;
    unsigned char b;
} rgb_struct;

/**
 * @brief main function and entry point of program.
 */
int main(int argc, char **argv)
{
    /* Step 1. struct data type.
    ------------------------------- */
    struct Student stu = {"Li", 2000, true}; // initialization
    strcpy(stu.name, "Wei");
    stu.born = 2000;
    stu.male = false;

    printf("Student %s, born in %d, gender %s\n",
           stu.name,
           stu.born,
           stu.male ? "male" : "female");

    struct Student students[100];
    students[50].born = 2002;

    /* Step 2. struct padding and memroy aligned.
    ----------------------------------------------- */
    std::cout << "Size of Student1: " << sizeof(Student1) << std::endl;
    std::cout << "Size of Student2: " << sizeof(Student2) << std::endl;

    /* Step 3. typedef in C++.
    ---------------------------- */
    // the following two lines are identical
    // unsigned char color[3] = {255, 0, 255};
    vec3b color = {255, 0, 255};
    std::cout << std::hex;
    std::cout << "R=" << +color[0] << ", ";
    std::cout << "G=" << +color[1] << ", ";
    std::cout << "B=" << +color[2] << std::endl;

    rgb_struct rgb = {0, 255, 128};
    std::cout << "R=" << +rgb.r << ", ";
    std::cout << "G=" << +rgb.g << ", ";
    std::cout << "B=" << +rgb.b << std::endl;

    std::cout << sizeof(rgb.r) << std::endl;
    std::cout << sizeof(+rgb.r) << std::endl; /* why 4? memory aligned */

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ struct.cpp
 * $ clang++ struct.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */