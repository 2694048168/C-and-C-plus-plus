/**
 * @file this_pointer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-13
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the this pointer(current object) of Class
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

    Student(const char *name, int born, bool male)
    {
        this->name =  new char[1024];
        this->setName(name);
        this->born = born;
        this->male = male;
        std::cout << "Constructor: Person(const char, int , bool)" << std::endl;
    }

    // default destructor
    ~Student()
    {
        std::cout << "To destory object: " << name << std::endl;
        delete[] name;
    }

    void setName(const char *name)
    {
        strncpy_s(this->name, 1024, name, 1024);
    }
    void setBorn(int born)
    {
        this->born = born;
    }

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

// size_t Student::student_total = 0; /* static variable definition it here */

int main(int argc, char const *argv[])
{
    Student *class1 = new Student[3]{
        {"Tom", 2000, true},
        {"Bob", 2001, true},
        {"Amy", 2002, false},
    };

    class1[1].printInfo();
    delete[] class1;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ this_pointer.cpp
 * $ clang++ this_pointer.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */