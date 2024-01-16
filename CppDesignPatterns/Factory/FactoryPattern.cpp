/**
 * @file FactoryPattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 工厂模式
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>

// 产品类的基类
class AbstractSmile
{
public:
    virtual void transform() = 0;
    virtual void ability()   = 0;

    // ! 利用到多态, 那么一定要实现一个抽象基类的析构函数
    virtual ~AbstractSmile() {}
};

class SheepSmile : public AbstractSmile
{
public:
    void transform() override
    {
        std::cout << "Sheep transform\n";
    }

    void ability() override
    {
        std::cout << "Sheep ability\n";
    }
};

class LionSmile : public AbstractSmile
{
public:
    void transform() override
    {
        std::cout << "Lion transform\n";
    }

    void ability() override
    {
        std::cout << "Lion ability\n";
    }
};

class BatSmile : public AbstractSmile
{
public:
    void transform() override
    {
        std::cout << "Bat transform\n";
    }

    void ability() override
    {
        std::cout << "Bat ability\n";
    }
};

// 定义工厂抽象基类
class AbstractFactory
{
public:
    virtual AbstractSmile *createSmile() = 0;

    // ! 利用到多态, 那么一定要实现一个抽象基类的析构函数
    virtual ~AbstractFactory() {}
};

// 生产Sheep工厂类
class SheepFactory : public AbstractFactory
{
public:
    AbstractSmile *createSmile() override
    {
        return new SheepSmile;
    }

    ~SheepFactory()
    {
        std::cout << "~SheepFactory\n";
    }
};

// 生产Lion工厂类
class LionFactory : public AbstractFactory
{
public:
    AbstractSmile *createSmile() override
    {
        return new LionSmile;
    }

    ~LionFactory()
    {
        std::cout << "~LionFactory\n";
    }
};

// 生产Bat工厂类
class BatFactory : public AbstractFactory
{
public:
    AbstractSmile *createSmile() override
    {
        return new BatSmile;
    }

    ~BatFactory()
    {
        std::cout << "~BatFactory\n";
    }
};

// ====================================
int main(int argc, const char **argv)
{
    AbstractFactory *pFactory = new SheepFactory;
    // AbstractFactory *pFactory = new LionFactory;
    // AbstractFactory *pFactory = new BatFactory;
    AbstractSmile   *pSmile   = pFactory->createSmile();

    pSmile->transform();
    pSmile->ability();

    if (pFactory != nullptr)
    {
        delete pFactory;
        pFactory = nullptr;
    }

    if (pSmile != nullptr)
    {
        delete pSmile;
        pSmile = nullptr;
    }

    return 0;
}
