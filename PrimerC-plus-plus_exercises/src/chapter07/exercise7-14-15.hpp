/* exercise 7-14
** 练习7.14: 编写一个构造函数，令其用提供的类内初始值显式地初始化成员
** solution: 
**
** 练习7.14: 为 Person 类添加正确的构造函数
**
*/

#ifndef EXERCISE7_14_15_H
#define EXERCISE7_14_15_H

#include <string>
#include <iostream>

struct Person;
std::istream &read(std::istream&, Person&);

struct Person 
{
    Person() = default;
    Person(const std::string sname, const std::string saddr):name(sname), address(saddr){ }
    Person(std::istream &is)
    { 
        read(is, *this); 
    }
    
    std::string getName() const { return name; }
    std::string getAddress() const { return address; }
    
    std::string name;
    std::string address;
};

std::istream &read(std::istream &is, Person &person)
{
    is >> person.name >> person.address;
    return is;
}

std::ostream &print(std::ostream &os, const Person &person)
{
    os << person.name << " " << person.address;
    return os;
}

#endif // EXERCISE7_14_15_H
