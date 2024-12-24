/**
 * @file Decorator.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Decorator.cpp -std=c++20
 * clang++ Decorator.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

class Component
{
public:
    virtual void operation() = 0;
};

class ConcreteComponent : public Component
{
public:
    void operation() override
    {
        std::cout << "From ConcreteComponent." << std::endl;
    }
};

class Decorator : public Component
{
public:
    Decorator(Component &component)
        : component_(component)
    {
    }

    void operation() override
    {
        std::cout << "from Decorator." << std::endl;
        component_.operation();
    }

private:
    Component &component_;
};

// Demo1: 自助冰淇淋制作机
// Component
class IceCream
{
public:
    virtual std::string getDescription() const = 0;
    virtual double      cost() const           = 0;
    virtual ~IceCream()                        = default;
};

// Concrete Component
class VanillaIceCream : public IceCream
{
public:
    std::string getDescription() const override
    {
        return "Vanilla Ice Cream";
    }

    double cost() const override
    {
        return 160.0;
    }
};

// Decorator
class DecoratorBase : public IceCream
{
protected:
    IceCream *iceCream;

public:
    DecoratorBase(IceCream *ic)
        : iceCream(ic)
    {
    }

    std::string getDescription() const override
    {
        return iceCream->getDescription();
    }

    double cost() const override
    {
        return iceCream->cost();
    }
};

// Concrete Decorator - adds chocolate topping.
class ChocolateDecorator : public DecoratorBase
{
public:
    ChocolateDecorator(IceCream *ic)
        : DecoratorBase(ic)
    {
    }

    std::string getDescription() const override
    {
        return iceCream->getDescription() + " with Chocolate";
    }

    double cost() const override
    {
        return iceCream->cost() + 100.0;
    }
};

// Concrete Decorator - adds caramel topping.
class CaramelDecorator : public DecoratorBase
{
public:
    CaramelDecorator(IceCream *ic)
        : DecoratorBase(ic)
    {
    }

    std::string getDescription() const override
    {
        return iceCream->getDescription() + " with Caramel";
    }

    double cost() const override
    {
        return iceCream->cost() + 150.0;
    }
};

// Demo2: 模拟的绘图组件
// Component
class Widget
{
public:
    virtual void draw() = 0;
};

// Concrete Component
class TextField : public Widget
{
private:
    int width;
    int height;

public:
    TextField(int w, int h)
        : width{w}
        , height{h}
    {
    }

    void draw() override
    {
        std::cout << "TextField: " << width << "," << height << '\n';
    }
};

// Decorator
class DecoratorBase_ : public Widget
{
private:
    Widget *wid;

public:
    DecoratorBase_(Widget *w)
        : wid{w}
    {
    }

    void draw() override
    {
        wid->draw();
    }
};

// Concrete Decorator
class BorderDecorator : public DecoratorBase_
{
public:
    BorderDecorator(Widget *w)
        : DecoratorBase_(w)
    {
    }

    void draw() override
    {
        //基础功能
        DecoratorBase_::draw();
        //扩展功能
        std::cout << " BorderDecorator\n";
    }
};

// Concrete Decorator
class ScrollDecorator : public DecoratorBase_
{
public:
    ScrollDecorator(Widget *w)
        : DecoratorBase_(w)
    {
    }

    /*virtual*/
    void draw() override
    {
        //基础功能
        DecoratorBase_::draw();
        //扩展功能
        std::cout << " ScrollDecorator\n";
    }
};

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "------------------------------------\n";
    ConcreteComponent concreteComponent;

    Decorator decoratedComponent(concreteComponent);

    decoratedComponent.operation();

    std::cout << "------------------------------------\n";
    // Create a vanilla ice cream
    IceCream *vanillaIceCream = new VanillaIceCream();
    std::cout << "Order: " << vanillaIceCream->getDescription() << ", Cost: Rs." << vanillaIceCream->cost()
              << std::endl;

    // Wrap it with ChocolateDecorator
    IceCream *chocolateIceCream = new ChocolateDecorator(vanillaIceCream);
    std::cout << "Order: " << chocolateIceCream->getDescription() << ", Cost: Rs." << chocolateIceCream->cost()
              << std::endl;

    // Wrap it with CaramelDecorator
    IceCream *caramelIceCream = new CaramelDecorator(chocolateIceCream);
    std::cout << "Order: " << caramelIceCream->getDescription() << ", Cost: Rs." << caramelIceCream->cost()
              << std::endl;

    delete vanillaIceCream;   /* standard OP. there just for test */
    delete chocolateIceCream; /* standard OP. there just for test */
    delete caramelIceCream;   /* standard OP. there just for test */

    std::cout << "------------------------------------\n";
    Widget *pWidget = new BorderDecorator(new ScrollDecorator(new TextField(80, 24)));
    pWidget->draw();

    return 0;
}
