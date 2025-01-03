/**
 * @file PrototypePattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ PrototypePattern.cpp -std=c++20
 * clang++ PrototypePattern.cpp -std=c++20
 * 
 */

#include <iostream>
#include <map>
#include <memory>
#include <string>

// Demo1
//定义原型基类
class Prototype
{
public:
    virtual ~Prototype() {}

    virtual Prototype *clone() const = 0;
    virtual void       print() const = 0;
};

//定义具体原型类
class ConcretePrototype : public Prototype
{
private:
    std::string name;

public:
    ConcretePrototype(const std::string &name)
        : name(name)
    {
    }

    Prototype *clone() const override
    {
        return new ConcretePrototype(*this);
    }

    void print() const override
    {
        std::cout << "Prototype: " << name << std::endl;
    }
};

// Demo2
class Prototype2
{
public:
    virtual ~Prototype2() {}

    virtual Prototype2 *clone() const = 0;
    virtual void        info() const  = 0;
};

class ConcretePrototypeA : public Prototype2
{
public:
    ConcretePrototypeA(int id, std::string name)
        : m_id(id)
        , m_name(name)
    {
    }

    Prototype2 *clone() const override
    {
        return new ConcretePrototypeA(*this);
    }

    void info() const override
    {
        std::cout << "ConcretePrototypeA: id = " << m_id << ", name = " << m_name << std::endl;
    }

private:
    int         m_id;
    std::string m_name;
};

class ConcretePrototypeB : public Prototype2
{
public:
    ConcretePrototypeB(std::string description)
        : m_description(description)
    {
    }

    Prototype2 *clone() const override
    {
        return new ConcretePrototypeB(*this);
    }

    void info() const override
    {
        std::cout << "ConcretePrototypeB: description = " << m_description << std::endl;
    }

private:
    std::string m_description;
};

// Demo1: 基于智能指针封装的原型模式
class PrototypeP
{
public:
    virtual std::unique_ptr<PrototypeP> clone() const = 0;

    void printValue() const
    {
        std::cout << "Origin Value." << std::endl;
    }

    virtual ~PrototypeP() = default;
};

class ConcretePrototypeP : public PrototypeP
{
private:
    int value;

public:
    ConcretePrototypeP(int v)
        : value(v)
    {
    }

    std::unique_ptr<PrototypeP> clone() const override
    {
        return std::make_unique<ConcretePrototypeP>(value);
    }

    void printValue() const
    {
        std::cout << "Value: " << value << std::endl;
    }
};

// Demo2: 基于工厂的方式管理各种原型
class Prototype_
{
public:
    int         data;
    std::string name;

    Prototype_(int data, const std::string &name)
        : data(data)
        , name(name)
    {
    }

    virtual Prototype_ *clone()
    {
        return new Prototype_(*this);
    }
};

class PrototypeFactory
{
private:
    std::map<std::string, Prototype_ *> prototypes;

public:
    PrototypeFactory()
    {
        prototypes["original"] = new Prototype_(40, "Original");
        prototypes["copy"]     = new Prototype_(100, "Copy");
    }

    Prototype_ *create(const std::string &type)
    {
        return prototypes[type]->clone();
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    //创建原型对象
    ConcretePrototype prototype("Original");
    //使用原型对象创建新对象
    Prototype        *clone = prototype.clone();
    clone->print();
    //释放内存
    delete clone;

    std::cout << "--------------------------------------\n";
    ConcretePrototypeA *prototypeA = new ConcretePrototypeA(1, "First");
    Prototype2         *cloneA     = prototypeA->clone();
    cloneA->info();
    ConcretePrototypeB *prototypeB = new ConcretePrototypeB("This is a prototype");
    Prototype2         *cloneB     = prototypeB->clone();
    cloneB->info();
    delete prototypeA;
    delete cloneA;
    delete prototypeB;
    delete cloneB;

    std::cout << "--------------------------------------\n";
    auto prototype_      = std::make_unique<ConcretePrototypeP>(5);
    auto clonedPrototype = prototype_->clone();
    prototype_->printValue();
    clonedPrototype->printValue();

    std::cout << "--------------------------------------\n";
    PrototypeFactory factory;
    Prototype_      *original = factory.create("original");
    Prototype_      *copy     = factory.create("copy");

    std::cout << "Original: data=" << original->data << ", name=" << original->name << std::endl;
    std::cout << "Copy: data=" << copy->data << ", name=" << copy->name << std::endl;

    delete original;
    delete copy;

    return 0;
}
