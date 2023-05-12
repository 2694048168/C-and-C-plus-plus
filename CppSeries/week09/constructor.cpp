/**
 * @file constructor.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-12
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief the constructor of Class in C++
 * @attention
 *
 */

#include <iostream>
#include <cstring>

class Student
{
private:
    char name[4];
    int born;
    bool male;

public:
    Student() /* default constuctor */
    {
        name[0] = 0;
        born = 0;
        male = false;
        std::cout << "Constructor: Person()" << std::endl;
    }

    // constructor and init member varialbes
    Student(const char *initName) : born(0), male(true)
    {
        setName(initName);
        std::cout << "Constructor: Person(const char*)" << std::endl;
    }
    // overload the constructor
    Student(const char *initName, int initBorn, bool isMale)
    {
        setName(initName);
        born = initBorn;
        male = isMale;
        std::cout << "Constructor: Person(const char, int , bool)" << std::endl;
    }

    void setName(const char *s)
    {
        strncpy(name, s, sizeof(name));
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
    Student yu;
    yu.printInfo();

    yu.setName("Yu");
    yu.setBorn(2000);
    yu.setGender(true);
    yu.printInfo();

    Student li("li");
    li.printInfo();

    Student xue = Student("weili", 1962, true);
    // a question: what will happen since "weili" has 4+ characters?
    // overflow and access random value(memory)
    xue.printInfo();

    // heap memory layout
    Student *zhou = new Student("Zhou", 1991, false);
    zhou->printInfo();
    delete zhou;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ constructor.cpp
 * $ clang++ constructor.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */