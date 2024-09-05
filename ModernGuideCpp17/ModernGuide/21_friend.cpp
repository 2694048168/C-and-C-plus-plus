/**
 * @file 21_friend.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-04
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** friend关键字在C++中是一个比较特别的存在,因为在大多数编程语言中是没有提供friend关键字的.
 * friend关键字用于声明类的友元,友元可以无视类中成员的属性(public,protected,private),
 * 友元类或友元函数都可以访问,这就完全破坏了面向对象编程中封装性的概念.
 * ?但有的时候,friend关键字确实会让程序猿少写很多代码,因此 friend 还是在很多程序中被使用到.
 * 
 * 1. 声明一个类为另外一个类的友元时,不再需要使用class关键字,并且还可以使用类的别名(使用 typedef 或者 using 定义);
 * 2. 为类模板声明友元
class Tom;
template<typename T>  
class Person
{
    friend T;
};

int main()
{
    Person<Tom> p;
    Person<int> pp;
    return 0;
}
 * *Tom类是Person类的友元;
 * *对于int类型的模板参数,友元声明被忽略;
 * 这样一来,就可以在模板实例化时才确定一个模板类是否有友元,以及谁是这个模板类的友元.
 * 
 */

#include <iostream>

// 类声明
class Tom;
// 定义别名
using Honey = Tom;

// 定义两个测试类
class Jack
{
    // 声明友元
    // friend class Tom;    // C++98 标准语法
    friend Tom;                // C++11 标准语法
    std::string name = "jack"; // 默认私有

    void print() // 默认私有
    {
        std::cout << "my name is " << name << std::endl;
    }
};

class Lucy
{
protected:
    // 声明友元
    // friend class Tom;    // C++98 标准语法
    friend Honey; // C++11 标准语法
    std::string name = "lucy";

    void print()
    {
        std::cout << "my name is " << name << std::endl;
    }
};

class Tom
{
public:
    void print()
    {
        // 通过类成员对象访问其私有成员
        std::cout << "invoke Jack private member: " << jObj.name << std::endl;
        std::cout << "invoke Jack private function: " << std::endl;
        jObj.print();

        std::cout << "invoke Lucy private member: " << lObj.name << std::endl;
        std::cout << "invoke Lucy private function: " << std::endl;
        lObj.print();
    }

private:
    std::string name = "tom";
    Jack        jObj;
    Lucy        lObj;
};

// ==========================
// 将其模板类型定义为了它们的友元(如果是模板类型是基础类型友元的定义就被忽略了)
template<typename T>
class Rectangle
{
public:
    friend T;

    Rectangle(int w, int h)
        : width(w)
        , height(h)
    {
    }

private:
    int width;
    int height;
};

template<typename T>
class Circle
{
public:
    friend T;

    Circle(int r)
        : radius(r)
    {
    }

private:
    int radius;
};

// 校验类
class Verify
{
public:
    void verifyRectangle(int w, int h, Rectangle<Verify> &r)
    {
        if (r.width >= w && r.height >= h)
        {
            std::cout << "矩形的宽度和高度满足条件!\n";
        }
        else
        {
            std::cout << "矩形的宽度和高度不满足条件!\n";
        }
    }

    void verifyCircle(int r, Circle<Verify> &c)
    {
        if (r >= c.radius)
        {
            std::cout << "圆形的半径满足条件!\n";
        }
        else
        {
            std::cout << "圆形的半径不满足条件!\n";
        }
    }
};

// ------------------------------------
int main(int argc, const char **argv)
{
    // Tom 类分别作为了Jack类和Lucy类的友元类,
    // 然后在Tom类中定义了Jack类和Lucy类的对象jObj和lObj,
    // 这样就可以在Tom类中通过这两个类对象直接访问它们各自的私有或者受保护的成员变量或者成员函数了.
    Tom t;
    t.print();

    // =======================
    std::cout << "=======================\n";
    Verify            v;
    Circle<Verify>    circle(30);
    Rectangle<Verify> rect(90, 100);
    v.verifyCircle(60, circle);
    v.verifyRectangle(100, 100, rect);

    return 0;
}
