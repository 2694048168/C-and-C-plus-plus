/**
 * @file 14_StaticPolymorphism.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-16
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

/**
* @brief CRTP（Curiously Recurring Template Pattern）是一种 C++ 编程技巧
* 使用模板类和继承的组合来实现静态多态, 该模式的关键思想是: 在模板类的定义中,
* 模板参数是当前类自身（通常是派生类）; 这个技巧通常用于实现编译时多态，优化性能
* 
* 使用了 static_cast 进行类型转换，根据 CRTP 的定义，
* 在 Info 的派生类中调用 getClassName 接口，并且 T 就是这里的派生类，
* 这里的 static_cast 转换一定是合法的，因为这里的 this 就是派生类型 T
* 
*/
template<typename T>
class Info
{
public:
    [[nodiscard]] std::string getClassName()
    {
        return static_cast<T *>(this)->getClassNameImpl();
    }
};

class C : public Info<C>
{
public:
    [[nodiscard]] std::string getClassNameImpl()
    {
        return "C";
    }
};

class D : public Info<D>
{
public:
    [[nodiscard]] std::string getClassNameImpl()
    {
        return "D";
    }
};

template<typename Derived>
class Printer
{
public:
    void print()
    {
        // 施展魔法，调用派生类的具体实现 ✨
        static_cast<Derived *>(this)->printImpl();
        // 每次打印后都来点花里胡哨的装饰 🎀
        std::cout << "=== 打印完成 ===\n";
    }

protected:
    void printImpl()
    {
        std::cout << "哎呀，这台打印机还没设置打印方式呢！😅\n";
    }
};

class ColorPrinter : public Printer<ColorPrinter>
{
public:
    void printImpl()
    {
        std::cout << "🎨 哇！我可以打印彩色的小花花！\n";
    }
};

class BWPrinter : public Printer<BWPrinter>
{
public:
    void printImpl()
    {
        std::cout << "⚫ 我是一本正经的黑白打印机～\n";
    }
};

template<typename Derived>
class Animal
{
public:
    void makeSound()
    {
        std::cout << "动物准备开口啦...\n";
        static_cast<Derived *>(this)->soundImpl();
        std::cout << "嗯！好响亮的叫声呢！🎵\n";
    }

    void findFood()
    {
        std::cout << "肚子咕咕叫，该觅食啦...\n";
        static_cast<Derived *>(this)->findFoodImpl();
    }

protected:
    void soundImpl()
    {
        std::cout << "（这只小可爱还在害羞呢~）😊\n";
    }

    void findFoodImpl()
    {
        std::cout << "（还不知道吃什么好...）🤔\n";
    }
};

class Cat : public Animal<Cat>
{
public:
    void soundImpl()
    {
        std::cout << "喵星人优雅地说：喵~ 铲屎官快来！🐱\n";
    }

    void findFoodImpl()
    {
        std::cout << "猫猫优雅地翻翻小鱼干，顺便打翻零食罐 🐟\n";
    }
};

class Duck : public Animal<Duck>
{
public:
    void soundImpl()
    {
        std::cout << "鸭鸭开心地嘎嘎嘎~🦆\n";
    }

    void findFoodImpl()
    {
        std::cout << "鸭鸭在池塘里快乐地捕鱼，顺便打个水漂 💦\n";
    }
};

template<typename Derived>
class Builder
{
public:
    // 每个积木块都会乖乖返回自己，方便下一块积木接上来 🧩
    Derived &name(const std::string &name)
    {
        std::cout << "给机器人起名字啦：" << name << " 🏷️" << std::endl;
        return static_cast<Derived &>(*this);
    }

    Derived &color(const std::string &color)
    {
        std::cout << "给机器人换新衣服：" << color << " 🎨" << std::endl;
        return static_cast<Derived &>(*this);
    }
};

// 这个小机器人制造商特别调皮，还能设置能量等级呢！
class RobotBuilder : public Builder<RobotBuilder>
{
public:
    RobotBuilder &power(int level)
    {
        std::cout << "给机器人充能量：" << level << " ⚡" << std::endl;
        return *this;
    }
};
