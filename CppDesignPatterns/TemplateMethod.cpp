/**
 * @file TemplateMethod.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-01
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ TemplateMethod.cpp -std=c++20
 * clang++ TemplateMethod.cpp -std=c++20
 * 
 */

#include <iostream>

class AbstractClass
{
public:
    //模板方法
    void templateMethod()
    {
        //算法步骤
        execute1();
        execute2();
    }

    virtual ~AbstractClass() = default;

protected:
    //基本操作方法
    virtual void execute1() = 0;
    virtual void execute2() = 0;
};

class ConcreteClassA : public AbstractClass
{
protected:
    void execute1() override
    {
        std::cout << "ConcreteClassA: execute1 called" << std::endl;
    }

    void execute2() override
    {
        std::cout << "ConcreteClassA: execute2 called" << std::endl;
    }
};

class ConcreteClassB : public AbstractClass
{
protected:
    void execute1() override
    {
        std::cout << "ConcreteClassB: execute1 called" << std::endl;
    }

    void execute2() override
    {
        std::cout << "ConcreteClassB: execute2 called" << std::endl;
    }
};

// 基于模板方法模式实现的模拟汽车生产
class VehicleTemplate
{
public:
    void buildVehicle()
    {
        assembleBody();
        installEngine();
        addWheels();
        std::cout << "Vehicle is ready!\n";
    }

    virtual ~VehicleTemplate() = default;

    virtual void assembleBody()  = 0;
    virtual void installEngine() = 0;
    virtual void addWheels()     = 0;
};

class Car : public VehicleTemplate
{
public:
    void assembleBody() override
    {
        std::cout << "Assembling car body.\n";
    }

    void installEngine() override
    {
        std::cout << "Installing car engine.\n";
    }

    void addWheels() override
    {
        std::cout << "Adding 4 wheels to the car.\n ";
    }
};

class Motorcycle : public VehicleTemplate
{
public:
    void assembleBody() override
    {
        std::cout << "Assembling motorcycle frame.\n";
    }

    void installEngine() override
    {
        std::cout << "Installing motorcycle engine.\n";
    }

    void addWheels() override
    {
        std::cout << "Adding 2 wheels to the motorcycle.\n ";
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    AbstractClass *classA = new ConcreteClassA();
    classA->templateMethod();
    AbstractClass *classB = new ConcreteClassB();
    classB->templateMethod();
    delete classA;
    delete classB;

    std::cout << "--------------------------------------\n";
    std::cout << "Building a Car : \n";
    Car car;
    car.buildVehicle();
    std::cout << "\nBuilding a Motorcycle : \n";
    Motorcycle motorcycle;
    motorcycle.buildVehicle();

    return 0;
}
