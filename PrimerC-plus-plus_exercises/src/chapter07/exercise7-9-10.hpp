/* exercise 7-9
** 练习7.9: 对于7.1.2节中练习的代码，添加读取和打印Person对象的操作
** solution: 
**
** 练习7.10: 在下面这条 if 语句中，条件部分的作用是什么？
** if (read(read(cin, data1), data2))
** solution: 因为read函数返回的类型是引用，可以返回结果可以作为实参被read函数继续使用。
** 连续读取两个对象data1, data2。如果读取成功，条件判断正确， 否则条件判断错误。
**
*/

#ifndef EXERCISE7_9_10_H
#define EXERCISE7_9_10_H

#include <string>
#include <iostream>

struct Person 
{
    std::string const& getName()    const { return name; }
    std::string const& getAddress() const { return address; }
    
    std::string name;
    std::string address;
};

std::istream &read(std::istream &is, Person &person)
{
    return is >> person.name >> person.address;
}

std::ostream &print(std::ostream &os, const Person &person)
{
    return os << person.name << " " << person.address;
}

#endif  // EXERCISE7_9_10_H