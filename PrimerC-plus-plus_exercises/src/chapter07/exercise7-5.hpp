/* exercise 7-5
** 练习7.5: 在 Person 类中提供一些操作使其能够返回姓名和地址
** 这些函数是否应该是 const 的呢？ 解释原因。
** solution: 
**
*/

#ifndef EXERCISE7_5_H
#define EXERCISE7_5_H

#include <string>

class Person 
{
    std::string name;
    std::string address;

public:
    auto get_name() const -> std::string const&
    { 
        return name; 
    }

    auto get_addr() const -> std::string const&
    { 
        return address; 
    }
};

#endif  // EXERCISE7_5_H