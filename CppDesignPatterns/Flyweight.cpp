/**
 * @file Flyweight.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-26
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Flyweight.cpp -std=c++20
 * clang++ Flyweight.cpp -std=c++20
 * 
 */

#include <iostream>
#include <map>
#include <memory>
#include <unordered_map>

class Flyweight
{
protected:
    int id; //内部状态

public:
    Flyweight(int id)
        : id(id)
    {
    }

    virtual void operation() const = 0;
};

class ConcreteFlyweightA : public Flyweight
{
public:
    ConcreteFlyweightA()
        : Flyweight(1)
    {
    }

    void operation() const override
    {
        std::cout << "Concrete Flyweight A, id: " << id << '\n';
    }
};

class ConcreteFlyweightB : public Flyweight
{
public:
    ConcreteFlyweightB()
        : Flyweight(2)
    {
    }

    void operation() const override
    {
        std::cout << "Concrete Flyweight B, id: " << id << '\n';
    }
};

// 定义享元工厂
class FlyweightFactory
{
private:
    std::unordered_map<int, std::shared_ptr<Flyweight>> flyweights;

public:
    FlyweightFactory() {}

    // 返回享元对象
    std::shared_ptr<Flyweight> getConcreteFlyweight(int id)
    {
        if (flyweights.find(id) == flyweights.end())
        {
            if (id % 2 == 0)
            {
                flyweights[id] = std::make_shared<ConcreteFlyweightA>();
            }
            else
            {
                flyweights[id] = std::make_shared<ConcreteFlyweightB>();
            }
        }
        return flyweights[id];
    }
};

// Demo: 模拟字符编辑器
// Flyweight
class Character
{
public:
    Character(char symbol)
        : symbol_(symbol)
    {
    }

    Character() = default;

    void print() const
    {
        std::cout << "Character: " << symbol_ << '\n';
    }

private:
    char symbol_;
};

// Flyweight factory
class CharacterFactory
{
public:
    static const Character &getCharacter(char symbol)
    {
        if (characters_.find(symbol) == characters_.end())
        {
            characters_[symbol] = Character(symbol);
            std::cout << "Create new char.\n";
        }
        return characters_[symbol];
    }

private:
    static std::map<char, Character> characters_;
};

std::map<char, Character> CharacterFactory::characters_;

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    FlyweightFactory factory;

    std::shared_ptr<Flyweight> f1 = factory.getConcreteFlyweight(1);
    std::shared_ptr<Flyweight> f2 = factory.getConcreteFlyweight(2);
    std::shared_ptr<Flyweight> f3 = factory.getConcreteFlyweight(3);

    f1->operation();
    f2->operation();
    f3->operation();

    std::cout << "=================================\n";
    const Character &A = CharacterFactory::getCharacter('A');
    const Character &B = CharacterFactory::getCharacter('B');

    // Reusing 'A'
    const Character &A2 = CharacterFactory::getCharacter('A');

    A.print();
    B.print();
    A2.print();

    return 0;
}
