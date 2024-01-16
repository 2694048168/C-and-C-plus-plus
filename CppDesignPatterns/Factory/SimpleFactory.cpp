/**
 * @file SimpleFactory.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 简单工厂模式
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

// 定义工厂类
enum class Type : char
{
    Sheep,
    Lion,
    Bat
};

class SmileFactory
{
public:
    AbstractSmile *createSmile(Type type)
    {
        AbstractSmile *pSmile = nullptr;
        switch (type)
        {
        case Type::Sheep:
            pSmile = new SheepSmile;
            break;
        case Type::Lion:
            pSmile = new LionSmile;
            break;
        case Type::Bat:
            pSmile = new BatSmile;
            break;
        default:
            break;
        }

        return pSmile;
    }
};

// ====================================
int main(int argc, const char **argv)
{
    SmileFactory  *pFactory = new SmileFactory;
    // AbstractSmile *pSmile   = pFactory->createSmile(Type::Lion);
    // AbstractSmile *pSmile = pFactory->createSmile(Type::Bat);
    AbstractSmile *pSmile   = pFactory->createSmile(Type::Sheep);

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
