/**
 * @file virtual.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief virtual function and pure-virtual function
 * @attention virtual table and polymorphic
 *
 */

#include <iostream>
#include <memory>
#include <string>

class Person
{
public:
    std::string name;

    Person(std::string n) : name(n) {}

    virtual void print()
    {
        std::cout << "Name: " << name << std::endl;
    }
};

class Person2
{
public:
    std::string name;

    Person2(std::string n) : name(n) {}

    virtual void print() = 0; /* pure-virtual function */
};

class Student : public Person
{
public:
    std::string id;

    Student(std::string n, std::string i) : Person(n), id(i) {}

    void print()
    {
        std::cout << "Name: " << name;
        std::cout << ". ID: " << id << std::endl;
    }
};

void printObjectInfo(Person &p)
{
    p.print();
}

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    {
        Student stu("yu", "2019");
        printObjectInfo(stu);
    }

    {
        Person *p = new Student("xue", "2020");
        p->print(); // if print() is not a virtual function, different output
        delete p;   // if its destructor is not virtual
    }

    { // if you want to call a function in the base class
        Student stu("li", "2021");
        stu.Person::print();

        Person *p = new Student("xue", "2020");
        p->Person::print();
        delete p;
    }

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ virtual.cpp
 * $ clang++ virtual.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */