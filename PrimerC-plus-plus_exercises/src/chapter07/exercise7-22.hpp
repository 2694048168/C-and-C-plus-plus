/* exercise 7-22
** 练习7.22: 修改你的 Person 类使其隐藏实现的细节
** solution: 
**
*/

#ifndef EXERCISE7_22_H
#define EXERCISE7_22_H

#include <string>
#include <iostream>

class Peason
{
public:
	Peason() = default;
	Peason(const std::string s1, const std::string s2) :name(s1), address(s2) { }
	Peason(std::istream &is);
	std::string get_name() const { return name; }
	std::string get_address() const { return address; }
private:
	std::string name;
	std::string address;
	friend std::istream &read(std::istream &is, Peason &item);
	friend std::ostream &print(std::ostream &os, const Peason &item);
};

std::istream &read(std::istream &is, Peason &item) 
{
	is >> item.name >> item.address;
	return is;
}

std::ostream &print(std::ostream &os, const Peason &item) 
{
	os << item.name << " " << item.address;
	return os;
}

Peason::Peason(std::istream &is) 
{
	read(is, *this);
}

#endif // EXERCISE7_22_H
