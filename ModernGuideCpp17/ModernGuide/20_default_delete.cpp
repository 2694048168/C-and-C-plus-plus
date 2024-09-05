/**
 * @file 20_default_delete.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-03
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/** 1. 类与默认函数
 * 在C++中声明自定义的类, 编译器会默认帮助程序员生成一些他们未自定义的成员函数,
 * 这样的函数版本被称为"默认函数", 这样的函数一共有六个:
 * *1. 无参构造函数：创建类对象;
 * *2. 拷贝构造函数：拷贝类对象;
 * *3. 移动构造函数：拷贝类对象;
 * *4. 拷贝赋值函数：类对象赋值;
 * *5. 移动赋值函数：类对象赋值;
 * *6. 析构函数：销毁类对象;
 * !在C++语法规则中,一旦程序员实现了这些函数的自定义版本,则编译器不会再为该类自动生成默认版本.
 * ?一旦声明了自定义版本的构造函数,则有可能导致定义的类型不再是POD类型,便不再能够享受POD类型为我们带来的便利.
 * 
 * 2. =default 和 =delete
 * 在C++11标准中称= default修饰的函数为显式默认【缺省】（explicit defaulted）函数,
 * 而称=delete修饰的函数为删除（deleted）函数或者显示删除函数.
 * *C++11引入显式默认和显式删除是为了增强对类默认函数的控制,让程序员能够更加精细地控制默认版本的函数.
 * 
 * 如果程序猿对C++类提供的默认函数(六个函数)进行了实现,那么可以通过 =default 将他们再次指定为默认函数.
 * !不能使用 =default 修饰这六个函数以外的函数.
 * ?但是可以使用 =delete 修饰类的其他任意成员函数.
 * 1. 禁止使用默认生成的函数
 * 2. 禁止使用自定义函数
 */

#include <iostream>

// =delete 表示显示删除, 显式删除可以避免用户使用一些不应该使用的类的成员函数,
// 使用这种方式可以有效的防止某些类型之间自动进行隐式类型转换产生的错误.
class Base
{
public:
    Base(int num)
        : m_num(num)
    {
    }

    Base(char c)       = delete;
    void print(char c) = delete;

    void print()
    {
        std::cout << "num: " << m_num << std::endl;
    }

    void print(int num)
    {
        std::cout << "num: " << num << std::endl;
    }

private:
    int m_num;
};

// --------------------------------------
int main(int argc, const char **argv)
{
    Base b(97); // 'a' 对应的 ascii 值为97

    // 禁用带 char类型参数的构造函数，防止隐式类型转换（char转int)
    // Base b1('a'); // !error
    b.print();
    b.print(97);

    // 禁止使用带char类型的自定义函数，防止隐式类型转换（char转int)
    // b.print('a'); // !error

    return 0;
}
