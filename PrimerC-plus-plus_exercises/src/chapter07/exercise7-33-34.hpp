/* exercise 7-33、7-34
** 练习7.33: 如果我们给Screen 添加一个如下所示的size成员将发生什么情况？如果出现了问题，请尝试修改它
**
** 练习7.34: 如果我们把第256页Screen类的pos的typedef放在类的最后一行会发生什么情况？
** solution:
** pos将是未定义的标志符，导致编译失败。
**
** 练习7.35: 解释下面代码的含义，说明其中的Type和initVal分别使用了哪个定义。如果代码存在错误，尝试修改它。
** 
*/

#ifndef EXERCISE7_33_34_H
#define EXERCISE7_33_34_H

#include <vector>
#include <string>
#include <iostream>

// solution: 7-33
pos Screen::size() const
{
    return height * width;
}

// 返回类型错误，在Screen类型外没有pos这样的数据类型。数据类型将未知。应改为：
Screen::pos Screen::size() const


// solution 7-35
typedef string Type;    //定义Type类型为string
Type initVal();    // 全局函数声明，返回类型为string

class Exercise
{
public:
    typedef double Type;    //定义Type类型为double
    Type setVal(Type);    //成员函数，返回类型，形参为double
    Type initVal();       //成员函数，返回类型为double（隐藏了同名的函数）
private:
    int val;
};

Type Exercise::setVal(Type parm)    //成员函数的定义，返回类型为string 参数类型为double
{
    val = parm + initVal();    //成员函数为initVal()
    return val;
}
//setVal函数返回值的类型和返回类型不符，改为：
Exercise::Type Exercise::setVal(Type parm)


#endif // EXERCISE7_33_34_H
