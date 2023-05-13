/**
 * @file const.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the const for variable and function member of Class
 * @attention const to function of class
 *
 */

#include <iostream>
#include <cstring>

class Student
{
private:
    const int BMI = 24;
    char *name;
    int born;
    bool male;

public:
    Student() /* default constuctor */
    {
        name = new char[1024]{0};
        born = 0;
        male = false;
        // BMI = 25; /* can not be modified */
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

    void setName(const char * name)
    {
        // strncpy(this->name, name, 1024);
        strncpy_s(this->name, 1024, name, 1024);
    }
    void setBorn(int born)
    {
        this->born = born;
    }
    int getBorn() const
    {
        // born++; /* const member function not modify the varialbe */
        return born;
    }
    // the declarations, the definitions are out of the class
    void setGender(bool isMale);
    void printInfo() const;
};

void Student::setGender(bool isMale)
{
    male = isMale;
}

void Student::printInfo() const
{
    std::cout << "Name: " << name << std::endl;
    std::cout << "Born in " << born << std::endl;
    std::cout << "Gender: " << (male ? "Male" : "Female") << std::endl;
}

int main(int argc, char const *argv[])
{
    Student li("Li", 1999, false);

    std::cout << "li.getBorn() = " << li.getBorn() << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ const.cpp
 * $ clang++ const.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */