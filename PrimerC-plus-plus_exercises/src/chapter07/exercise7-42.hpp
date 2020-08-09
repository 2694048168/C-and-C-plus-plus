/* exercise 7-42
** 练习7.42: 对于你在练习7.40 （参见7.5.1节，第261 页）中编写的类，确定哪些构造函数可以使用委托。
** 如果可以的话，编写委托构造函数。
** 如果不可以，从抽象概念列表中重新选择一个你认为可以使用委托构造函数的，为挑选出的这个概念编写类定义
**
*/

#ifndef EXERCISE7_42_H
#define EXERCISE7_42_H

#include <iostream>

class Book
{
private:
	std::string Name, ISBN, Author, Publisher;
	double Price = 0;
public:
	Book(const std::string & n, const std::string &I, double pr, const std::string & a, const std::string & p)
		: Name(n), ISBN(I), Price(pr), Author(a), Publisher(p) { }
	Book() : Book("", "", 0, "", "") { }
	Book(std::istream & is) : Book() { is >> *this; }
};


#endif // EXERCISE7_42_H
