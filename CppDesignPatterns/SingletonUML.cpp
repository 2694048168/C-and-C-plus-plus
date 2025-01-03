/**
 * @file SingletonUML.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ SingletonUML.cpp -std=c++20
 * clang++ SingletonUML.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

class Singleton
{
private:
    static bool       instanceFlag;
    static Singleton *single;

    Singleton()
    {
        printf("Private constructor finished.\n");
    }

public:
    static Singleton *getInstance();
    void              method();

    ~Singleton()
    {
        printf("Public de-constructor finished.\n");
        instanceFlag = false;
    }
};

bool       Singleton::instanceFlag = false;
Singleton *Singleton::single       = NULL;

Singleton *Singleton::getInstance()
{
    if (!instanceFlag)
    {
        single       = new Singleton();
        instanceFlag = true;
        return single;
    }
    else
    {
        return single;
    }
}

void Singleton::method()
{
    std::cout << "Method of the singleton class\n";
}

// Demo2: 增加析构函数
class Singleton_
{
public:
    static Singleton_ &getInstance()
    {
        //如果对象实例不存在就创建一个
        if (!instance)
        {
            instance = new Singleton_();
        }
        return *instance;
    }

    //给外部调用的接口
    void Operation()
    {
        std::cout << "Singleton is performing some operation." << std::endl;
    }

    Singleton_(const Singleton_ &)            = delete;
    Singleton_ &operator=(const Singleton_ &) = delete;

private:
    //私有化的构造函数
    Singleton_()
    {
        std::cout << "Singleton instance created." << std::endl;
    }

    //私有化的析构函数
    ~Singleton_()
    {
        std::cout << "Singleton instance destroyed." << std::endl;
    }

    static Singleton_ *instance;
};

Singleton_ *Singleton_::instance = nullptr;

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    Singleton *sc1, *sc2;

    sc1 = Singleton::getInstance();
    sc1->method();
    sc2 = Singleton::getInstance();
    sc2->method();

    delete sc1, sc2;

    std::cout << "--------------------------------------\n";
    Singleton_ &singleton = Singleton_::getInstance();
    singleton.Operation();

    return 0;
}
