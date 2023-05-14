/**
 * @file rtti.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief RTTI mechanism in C++
 * @attention
 *
 */

#include <iostream>
#include <string>
#include <typeinfo>

class Person
{
protected:
    std::string name;

public:
    Person(std::string name = "") : name(name){};

    virtual ~Person() {} /* must be virtual function for destruction */

    std::string getInfo() { return name; }
};

class Student : public Person
{
    std::string studentid;

public:
    Student(std::string name = "", std::string sid = "") : Person(name), studentid(sid){};

    std::string getInfo() { return name + ":" + studentid; }
};

/**
 * @brief main function
 */
int main(int argc, char const *argv[])
{
    Person person("Yu");
    Student student("Sam", "20210212");
    Person *pp = &student;
    Person &rp = student;

    Student *ps = (Student *)&person; // danger!

    std::cout << person.getInfo() << std::endl;
    std::cout << ps->getInfo() << std::endl; // danger if getInfo is not virtual

    ps = dynamic_cast<Student *>(&person);
    printf("address = %p\n", ps);
    pp = dynamic_cast<Person *>(&student);
    printf("address = %p\n", pp);

    /* ------------ typeid -------------- */
    std::cout << "------------ typeid --------------\n";
    std::string s("hello");

    std::cout << "typeid.name of s           is " << typeid(s).name()
              << std::endl;
    std::cout << "typeid.name of std::string is " << typeid(std::string).name()
              << std::endl;
    std::cout << "typeid.name of Student     is " << typeid(Student).name()
              << std::endl;

    if (typeid(std::string) == typeid(s))
        std::cout << "s is a std::string object.\n";

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ rtti.cpp
 * $ clang++ rtti.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */