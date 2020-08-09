/* exercise 7-24
** 练习7.24: 给你的Screen类添加三个构造函数：一个默认构造函数；
** 另一个构造函数接受宽和高的值，然后将contents初始化成给定数量的空白；
** 第三个构造函数接受宽和高的值以及一个字符，该字符作为初始化后屏幕的内容
** solution: 
**
** 练习7.25: Screen 能安全地依赖于拷贝和赋值操作的默认版本吗？如果能，为什么？如果不能？为什么？
** solution:
** Screen 类中的四个成员对象都是内置类型，因此能够安全地依赖于拷贝和赋值操作的默认版本.
**
*/

#ifndef EXERCISE7_24_25_H
#define EXERCISE7_24_25_H

#include <string>

class Screen 
{
    public:
        using pos = std::string::size_type;

        Screen() = default; // 1
        Screen(pos ht, pos wd):height(ht), width(wd), contents(ht*wd, ' '){ } // 2
        Screen(pos ht, pos wd, char c):height(ht), width(wd), contents(ht*wd, c){ } // 3

        char get() const { return contents[cursor]; }
        char get(pos r, pos c) const { return contents[r*width+c]; }

    private:
        pos cursor = 0;
        pos height = 0, width = 0;
        std::string contents;
};


#endif // EXERCISE7_24-25_H
