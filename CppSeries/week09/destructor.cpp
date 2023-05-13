/**
 * @file destructor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the destructor of Class in C++
 * @attention
 *
 */

#include <iostream>
#include <cstring>

class Student
{
private:
    char *name;
    int born;
    bool male;

public:
    Student() /* default constuctor */
    {
        name = new char[1024]{0};
        born = 0;
        male = false;
        std::cout << "Constructor: Person()" << std::endl;
    }

    Student(const char *initName, int initBorn, bool isMale)
    {
        name = new char[1024];
        setName(initName);
        born = initBorn;
        male = isMale;
        std::cout << "Constructor: Person(const char, int , bool)" << std::endl;
    }

    // default destructor
    ~Student()
    {
        std::cout << "To destory object: " << name << std::endl;
        delete[] name;
    }

    void setName(const char *s)
    {
        strncpy_s(name, 1024, s, 1024);
    }
    void setBorn(int b)
    {
        born = b;
    }
    // the declarations, the definitions are out of the class
    void setGender(bool isMale);
    void printInfo();
};

void Student::setGender(bool isMale)
{
    male = isMale;
}

void Student::printInfo()
{
    std::cout << "Name: " << name << std::endl;
    std::cout << "Born in " << born << std::endl;
    std::cout << "Gender: " << (male ? "Male" : "Female") << std::endl;
}

int main(int argc, char const *argv[])
{
    {
        Student yu;
        yu.printInfo();

        yu.setName("Yu");
        yu.setBorn(2000);
        yu.setGender(true);
        yu.printInfo();
    }
    Student xue = Student("XueQikun", 1962, true);
    xue.printInfo();

    Student *zhou = new Student("Zhou", 1991, false);
    zhou->printInfo();
    delete zhou;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ destructor.cpp
 * $ clang++ destructor.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */