/**
 * @file BuilderPattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2025-01-03
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ BuilderPattern.cpp -std=c++20
 * clang++ BuilderPattern.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

// Demo1: 没有Director参与
// Product
class Product
{
public:
    void setPart1(int part1)
    {
        part1_ = part1;
    }

    void setPart2(const std::string &part2)
    {
        part2_ = part2;
    }

    void show()
    {
        std::cout << "Part 1: " << part1_ << "\nPart 2: " << part2_ << std::endl;
    }

private:
    int         part1_;
    std::string part2_;
};

// Builder
class Builder
{
public:
    virtual void    buildPart1() = 0;
    virtual void    buildPart2() = 0;
    virtual Product getResult()  = 0;
};

// Concrete Builder
class ConcreteBuilder : public Builder
{
public:
    ConcreteBuilder()
    {
        product_ = new Product();
    }

    void buildPart1() override
    {
        product_->setPart1(30);
    }

    void buildPart2() override
    {
        product_->setPart2("Part2 !");
    }

    Product getResult() override
    {
        return *product_;
    }

private:
    Product *product_;
};

// Demo2: 包含Director
// Product
class Product_
{
public:
    void setPartA(const std::string &partA)
    {
        partA_ = partA;
    }

    void setPartB(const std::string &partB)
    {
        partB_ = partB;
    }

    void setPartC(const std::string &partC)
    {
        partC_ = partC;
    }

    void show()
    {
        std::cout << "Part A: " << partA_ << std::endl;
        std::cout << "Part B: " << partB_ << std::endl;
        std::cout << "Part C: " << partC_ << std::endl;
    }

private:
    std::string partA_;
    std::string partB_;
    std::string partC_;
};

// Builder
class Builder_
{
public:
    virtual void      buildPartA() = 0;
    virtual void      buildPartB() = 0;
    virtual void      buildPartC() = 0;
    virtual Product_ *getProduct() = 0;
};

// Concrete Builder
class ConcreteBuilder_ : public Builder_
{
public:
    ConcreteBuilder_()
    {
        product_ = new Product_();
    }

    void buildPartA()
    {
        product_->setPartA("Part A");
    }

    void buildPartB()
    {
        product_->setPartB("Part B");
    }

    void buildPartC()
    {
        product_->setPartC("Part C");
    }

    Product_ *getProduct()
    {
        return product_;
    }

private:
    Product_ *product_;
};

// Director
class Director
{
public:
    void construct(Builder_ *builder)
    {
        builder->buildPartA();
        builder->buildPartB();
        builder->buildPartC();
    }
};

// Demo1: 模拟汉堡的制作过程
class Burger
{
public:
    void setBurgerType(const std::string &type)
    {
        m_burgerType = type;
    }

    void setCheese(const bool cheese)
    {
        m_cheese = cheese;
    }

    void setPickles(const bool pickles)
    {
        m_pickles = pickles;
    }

    void setMayonnaise(const bool mayonnaise)
    {
        m_mayonnaise = mayonnaise;
    }

    std::string getBurgerType() const
    {
        return m_burgerType;
    }

    std::string getCheese() const
    {
        return m_cheese ? "Cheese" : "No cheese";
    }

    std::string getPickles() const
    {
        return m_pickles ? "Pickles" : "No pickles";
    }

    std::string getMayonnaise() const
    {
        return m_mayonnaise ? "Mayonnaise" : "No mayonnaise";
    }

private:
    std::string m_burgerType;
    bool        m_cheese;
    bool        m_pickles;
    bool        m_mayonnaise;
};

class BurgerBuilder
{
public:
    BurgerBuilder()
    {
        m_burger = new Burger();
    }

    BurgerBuilder *setBurgerType(const std::string &type)
    {
        m_burger->setBurgerType(type);
        return this;
    }

    BurgerBuilder *addCheese()
    {
        m_burger->setCheese(true);
        return this;
    }

    BurgerBuilder *addPickles()
    {
        m_burger->setPickles(true);
        return this;
    }

    BurgerBuilder *addMayonnaise()
    {
        m_burger->setMayonnaise(true);
        return this;
    }

    Burger *build()
    {
        return m_burger;
    }

private:
    Burger *m_burger;
};

// Demo2: 模拟台式机的组装过程
// Product
class Computer
{
public:
    void setCPU(const std::string &cpu)
    {
        cpu_ = cpu;
    }

    void setRAM(const std::string &ram)
    {
        ram_ = ram;
    }

    void setStorage(const std::string &storage)
    {
        storage_ = storage;
    }

    void displayInfo() const
    {
        std::cout << "Computer Configuration:"
                  << "\nCPU: " << cpu_ << "\nRAM: " << ram_ << "\nStorage: " << storage_ << "\n\n";
    }

private:
    std::string cpu_;
    std::string ram_;
    std::string storage_;
};

// Builder
class BuilderComputer
{
public:
    virtual void     buildCPU()     = 0;
    virtual void     buildRAM()     = 0;
    virtual void     buildStorage() = 0;
    virtual Computer getResult()    = 0;
};

// ConcreteBuilder
class GamingComputerBuilder : public BuilderComputer
{
private:
    Computer computer_;

public:
    void buildCPU() override
    {
        computer_.setCPU("Gaming CPU");
    }

    void buildRAM() override
    {
        computer_.setRAM("16GB DDR4");
    }

    void buildStorage() override
    {
        computer_.setStorage("1TB SSD");
    }

    Computer getResult() override
    {
        return computer_;
    }
};

// Director
class ComputerDirector
{
public:
    void construct(BuilderComputer &builder)
    {
        builder.buildCPU();
        builder.buildRAM();
        builder.buildStorage();
    }
};

// --------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "--------------------------------------\n";
    ConcreteBuilder builder;
    builder.buildPart1();
    builder.buildPart2();

    Product product = builder.getResult();
    product.show();

    std::cout << "--------------------------------------\n";
    ConcreteBuilder_ builder_;
    Director         director;
    director.construct(&builder_);
    Product_ *product_ = builder_.getProduct();
    product_->show();
    delete product_;

    std::cout << "--------------------------------------\n";
    Burger *burger = BurgerBuilder().setBurgerType("Chicken Burger")->addCheese()->addPickles()->build();
    std::cout << "Burger Type: " << burger->getBurgerType() << std::endl;
    std::cout << "Cheese: " << burger->getCheese() << std::endl;
    std::cout << "Pickles: " << burger->getPickles() << std::endl;
    std::cout << "Mayonnaise: " << burger->getMayonnaise() << std::endl;
    delete burger;

    std::cout << "--------------------------------------\n";
    GamingComputerBuilder gamingBuilder;
    ComputerDirector      director_;
    director_.construct(gamingBuilder);
    Computer gamingComputer = gamingBuilder.getResult();
    gamingComputer.displayInfo();

    return 0;
}
