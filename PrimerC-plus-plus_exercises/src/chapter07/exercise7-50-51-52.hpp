/* exercise 7-50、7-51、7-52
** 练习7.50: 确定在你的Person 类中是否有一些构造函数应该是explicit 的。
** solution:
** 将构造函数Person(std::istream & is)定义为explicit。
**
** 练习7.51: vector 将其单参数的构造函数定义成explicit 的，而string 则不是，你觉得原因何在？
** solution：
** 如果vector单参数构造函数不是explicit的，那么对于这样的一个函数void fun(vector v)来说，可以直接以这样的形式进行调用fun(5)，这种调用容易引起歧义，无法得知实参5指的是vector的元素个数还是只有一个值为5的元素。
** 而string类型不是一个容器，不存在这样的歧义问题。
**
** 练习7.52: 使用2.6.1节（第64页）的 Sales_data 类，解释下面的初始化过程。如果存在问题，尝试修改它。
** Sales_data item = {"987-0590353403", 25, 15.99};
** 
** solution:
** 将Sales_data的bookNo成员初始化为"978-0590353403"，将units_sold初始化为25，将revenue初始化为15.99
** 
*/

#ifndef EXERCISE7_50_51_52_H
#define EXERCISE7_50_51_52_H

// solution 7-50
#include <string>
#include <iostream>

struct Person 
{
    friend std::istream &read(std::istream &is, Person &person);
    friend std::ostream &print(std::ostream &os, const Person &person);

public:
    Person() = default;
    Person(const std::string sname, const std::string saddr):name(sname), address(saddr){ }
    explicit Person(std::istream &is){ read(is, *this); }

    std::string getName() const { return name; }
    std::string getAddress() const { return address; }
private:
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



#endif // EXERCISE7_50_51_52_H
