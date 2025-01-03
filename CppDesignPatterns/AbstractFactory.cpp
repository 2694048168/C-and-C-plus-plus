/**
 * @file AbstractFactory.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ AbstractFactory.cpp -std=c++20
 * clang++ AbstractFactory.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

// Abstract Product A
class AbstractProductA
{
public:
    virtual void operationA()   = 0;
    virtual ~AbstractProductA() = default;
};

// Concrete Product A1
class ProductA1 : public AbstractProductA
{
public:
    void operationA() override
    {
        std::cout << "Product A1 operation\n";
    }
};

// Concrete Product A2
class ProductA2 : public AbstractProductA
{
public:
    void operationA() override
    {
        std::cout << "Product A2 operation\n";
    }
};

// Abstract Product B
class AbstractProductB
{
public:
    virtual void operationB()   = 0;
    virtual ~AbstractProductB() = default;
};

// Concrete Product B1
class ProductB1 : public AbstractProductB
{
public:
    void operationB() override
    {
        std::cout << "Product B1 operation\n";
    }
};

// Concrete Product B2
class ProductB2 : public AbstractProductB
{
public:
    void operationB() override
    {
        std::cout << "Product B2 operation\n";
    }
};

// Abstract Factory
class AbstractFactory
{
public:
    virtual AbstractProductA *createProductA() = 0;
    virtual AbstractProductB *createProductB() = 0;
    virtual ~AbstractFactory()                 = default;
};

// Concrete Factory 1
class ConcreteFactory1 : public AbstractFactory
{
public:
    AbstractProductA *createProductA() override
    {
        return new ProductA1();
    }

    AbstractProductB *createProductB() override
    {
        return new ProductB1();
    }
};

// Concrete Factory 2
class ConcreteFactory2 : public AbstractFactory
{
public:
    AbstractProductA *createProductA() override
    {
        return new ProductA2();
    }

    AbstractProductB *createProductB() override
    {
        return new ProductB2();
    }
};

// 模拟披萨的制作过程
class Pizza
{
public:
    virtual void bake() = 0;
    virtual void cut()  = 0;
    virtual void box()  = 0;
    virtual ~Pizza()    = default;
};

class NewYorkCheesePizza : public Pizza
{
public:
    void bake() override
    {
        std::cout << "Baking New York-style cheese pizza." << std::endl;
    }

    void cut() override
    {
        std::cout << "Cutting New York-style cheese pizza." << std::endl;
    }

    void box() override
    {
        std::cout << "Boxing New York-style cheese pizza." << std::endl;
    }
};

class NewYorkPepperoniPizza : public Pizza
{
public:
    void bake() override
    {
        std::cout << "Baking New York-style pepperoni pizza." << std::endl;
    }

    void cut() override
    {
        std::cout << "Cutting New York-style pepperoni pizza." << std::endl;
    }

    void box() override
    {
        std::cout << "Boxing New York-style pepperoni pizza." << std::endl;
    }
};

class ChicagoCheesePizza : public Pizza
{
public:
    void bake() override
    {
        std::cout << "Baking Chicago-style cheese pizza." << std::endl;
    }

    void cut() override
    {
        std::cout << "Cutting Chicago-style cheese pizza." << std::endl;
    }

    void box() override
    {
        std::cout << "Boxing Chicago-style cheese pizza." << std::endl;
    }
};

class ChicagoPepperoniPizza : public Pizza
{
public:
    void bake() override
    {
        std::cout << "Baking Chicago-style pepperoni pizza." << std::endl;
    }

    void cut() override
    {
        std::cout << "Cutting Chicago-style pepperoni pizza." << std::endl;
    }

    void box() override
    {
        std::cout << "Boxing Chicago-style pepperoni pizza." << std::endl;
    }
};

class PizzaFactory
{
public:
    virtual Pizza *createCheesePizza()    = 0;
    virtual Pizza *createPepperoniPizza() = 0;
    virtual ~PizzaFactory()               = default;
};

class NewYorkPizzaFactory : public PizzaFactory
{
public:
    Pizza *createCheesePizza() override
    {
        return new NewYorkCheesePizza();
    }

    Pizza *createPepperoniPizza() override
    {
        return new NewYorkPepperoniPizza();
    }
};

class ChicagoPizzaFactory : public PizzaFactory
{
public:
    Pizza *createCheesePizza() override
    {
        return new ChicagoCheesePizza();
    }

    Pizza *createPepperoniPizza() override
    {
        return new ChicagoPepperoniPizza();
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    AbstractFactory  *factory1  = new ConcreteFactory1();
    AbstractProductA *productA1 = factory1->createProductA();
    AbstractProductB *productB1 = factory1->createProductB();
    productA1->operationA();
    productB1->operationB();
    AbstractFactory  *factory2  = new ConcreteFactory2();
    AbstractProductA *productA2 = factory2->createProductA();
    AbstractProductB *productB2 = factory2->createProductB();
    productA2->operationA();
    productB2->operationB();
    delete factory1;
    delete factory2;
    delete productA1;
    delete productB1;
    delete productA2;
    delete productB2;

    std::cout << "--------------------------------------\n";
    PizzaFactory *newYorkFactory        = new NewYorkPizzaFactory();
    Pizza        *newYorkCheesePizza    = newYorkFactory->createCheesePizza();
    Pizza        *newYorkPepperoniPizza = newYorkFactory->createPepperoniPizza();

    PizzaFactory *chicagoFactory        = new ChicagoPizzaFactory();
    Pizza        *chicagoCheesePizza    = chicagoFactory->createCheesePizza();
    Pizza        *chicagoPepperoniPizza = chicagoFactory->createPepperoniPizza();

    newYorkCheesePizza->bake();
    newYorkCheesePizza->cut();
    newYorkCheesePizza->box();

    newYorkPepperoniPizza->bake();
    newYorkPepperoniPizza->cut();
    newYorkPepperoniPizza->box();

    chicagoCheesePizza->bake();
    chicagoCheesePizza->cut();
    chicagoCheesePizza->box();

    chicagoPepperoniPizza->bake();
    chicagoPepperoniPizza->cut();
    chicagoPepperoniPizza->box();

    delete newYorkFactory;
    delete newYorkCheesePizza;
    delete newYorkPepperoniPizza;
    delete chicagoFactory;
    delete chicagoCheesePizza;
    delete chicagoPepperoniPizza;

    return 0;
}
