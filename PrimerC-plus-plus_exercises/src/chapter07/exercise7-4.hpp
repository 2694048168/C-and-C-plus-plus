/* exercise 7-4
** 练习7.4: 编写一个名为 Person 的类，使其表示人员的姓名和地址
** 使用 string 对象存放这些元素，接下来的练习将不断充实这个类的其他特征
** solution: 
**
*/

#ifndef EXERCISE7_4_H
#define EXERCISE7_4_H

#include <string>

class Person 
{
    std::string name;
    std::string address;
};

#endif  // EXERCISE7_4_H