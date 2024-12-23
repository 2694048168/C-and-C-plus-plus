/**
 * @file Bridge.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-23
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Bridge.cpp -std=c++20
 * clang++ Bridge.cpp -std=c++20
 */

#include <iostream>
#include <string>

class Implementation
{
public:
    virtual ~Implementation() {}

    virtual std::string newOperation() const = 0;
};

class ConcreteImplementationA : public Implementation
{
public:
    std::string newOperation() const override
    {
        return "ConcreteImplementationA: Here's the result on the platform A.\n";
    }
};

class ConcreteImplementationB : public Implementation
{
public:
    std::string newOperation() const override
    {
        return "ConcreteImplementationB: Here's the result on the platform B.\n";
    }
};

class Abstraction
{
protected:
    Implementation *implementation_;

public:
    Abstraction(Implementation *implementation)
        : implementation_(implementation)
    {
    }

    virtual ~Abstraction() {}

    virtual std::string doOperation() const
    {
        return "Abstraction: Base operation with:\n" + this->implementation_->newOperation();
    }
};

class RefinedAbstraction : public Abstraction
{
public:
    RefinedAbstraction(Implementation *implementation)
        : Abstraction(implementation)
    {
    }

    std::string doOperation() const override
    {
        return "RefinedAbstraction: Extended operation with:\n" + this->implementation_->newOperation();
    }
};

void ClientCode(const Abstraction &abstraction)
{
    std::cout << abstraction.doOperation();
}

// 生产不同颜色和不同车型的汽车
class IColor
{
public:
    virtual std::string Color() = 0;
    virtual ~IColor()           = default;
};

class RedColor : public IColor
{
public:
    std::string Color() override
    {
        return "of Red Color";
    }
};

class BlueColor : public IColor
{
public:
    std::string Color() override
    {
        return "of Blue Color";
    }
};

class ICarModel
{
public:
    virtual std::string WhatIsMyType() = 0;
    virtual ~ICarModel()               = default;
};

class Model_A : public ICarModel
{
    IColor *_myColor;

public:
    Model_A(IColor *obj)
        : _myColor(obj)
    {
    }

    std::string WhatIsMyType() override
    {
        return "I am a Model_A " + _myColor->Color();
    }
};

class Model_B : public ICarModel
{
    IColor *_myColor;

public:
    Model_B(IColor *obj)
        : _myColor(obj)
    {
    }

    std::string WhatIsMyType() override
    {
        return "I am a Model_B " + _myColor->Color();
        ;
    }
};

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "================================\n";
    Implementation *implementation_1 = new ConcreteImplementationA;
    Abstraction    *abstraction_1    = new Abstraction(implementation_1);

    ClientCode(*abstraction_1);
    std::cout << std::endl;

    if (implementation_1)
    {
        delete implementation_1;
        implementation_1 = nullptr;
    }
    if (abstraction_1)
    {
        delete abstraction_1;
        abstraction_1 = nullptr;
    }

    Implementation *implementation_2 = new ConcreteImplementationB;
    Abstraction    *abstraction_2    = new RefinedAbstraction(implementation_2);
    ClientCode(*abstraction_2);

    if (implementation_2)
    {
        delete implementation_2;
        implementation_2 = nullptr;
    }
    if (abstraction_2)
    {
        delete abstraction_2;
        abstraction_2 = nullptr;
    }

    std::cout << "================================\n";
    IColor *red  = new RedColor();
    IColor *blue = new BlueColor();

    ICarModel *modelA = new Model_B(red);
    ICarModel *modelB = new Model_A(blue);

    std::cout << "\n" << modelA->WhatIsMyType();
    std::cout << "\n" << modelB->WhatIsMyType() << "\n\n";

    if (red)
    {
        delete red;
        red = nullptr;
    }
    if (blue)
    {
        delete blue;
        blue = nullptr;
    }
    if (modelA)
    {
        delete modelA;
        modelA = nullptr;
    }
    if (modelB)
    {
        delete modelB;
        modelB = nullptr;
    }

    return 0;
}
