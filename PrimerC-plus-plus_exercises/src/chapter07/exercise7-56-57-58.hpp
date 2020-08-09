/* exercise 7-56、7-57、7-58
** 练习7.56: 什么是类的静态成员？它有何优点？静态成员与普通成员有何区别？
** solution：
** 静态成员是指声明语句之前带有关键字static的类成员，
** 静态成员不是任意单独对象的组成部分，而是由该类的全体对象所共享。
**
** 静态成员的优点包括：作用域位于类的范围之内，避免与其他类的成员或者全局作用域的名字冲突；
**                   可以是私有成员，而全局对象不可以；
** 通过阅读程序可以非常容易地看出静态成员与特定类关联，使得程序的含义清晰明了。
** 静态成员与普通成员的区别主要体现在普通成员与类的对象关联，是某个具体对象的组成部分；
** 而静态成员不从属于任何具体的对象，它由该类的所有对象共享。
** 另外，还有一个细微的区别，静态成员可以作为默认参数，而普通成员不能作为默认参数。
**
** 练习7.57: 编写你自己的Account 类
** solution：
**
** 练习7.58: 下面的静态数据成员的声明和定义有错误吗？请解释原因。
//example.h
class Example{
public:
    static double rate = 6.5;     //不是字面值常量类型的常量表达式的静态数据成员不允许在类内初始化
    //正确，但是在类外应该在定义一下，比如：const int Example::vetSize
    static const int vecSize = 20;
    //错误，constexpr static数据成员必须是字面值类型，vector非字面值类型，不允许类内初始化
    static vector<double> vec(vecSize);
};
//example.C
#include “example.h”           //两者在上面都错了，需要重新给出初始值
double Example::rate;
vector<double> Example::vec;
** 
*/

#ifndef EXERCISE7_56_57_58_H
#define EXERCISE7_56_57_58_H

// solution 7-57

#include <string>

class Account 
{
public:
    void calculate() { amount += amount * interestRate; }
    static double rate() { return interestRate; }
    static void rate(double newRate) { interestRate = newRate; }
    
private:
    std::string owner;
    double amount;
    static double interestRate;
    static constexpr double todayRate = 42.42;
    static double initRate() { return todayRate; }
};

double Account::interestRate = initRate();


#endif // EXERCISE7_56_57_58_H
