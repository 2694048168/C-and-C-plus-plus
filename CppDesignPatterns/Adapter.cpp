/**
 * @file Adapter.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-23
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Adapter.cpp -std=c++20
 * clang++ Adapter.cpp -std=c++20
 * 
 */

#include <functional>
#include <iostream>

// 目标接口
class Target
{
public:
    virtual void request() = 0;
};

// 源接口
class Adoptee
{
public:
    void specificRequest()
    {
        std::cout << "Adoptee specific request\n";
    }
};

// 被适配后的源接口
class Adapter : public Target
{
public:
    Adapter(Adoptee *adoptee)
        : m_adoptee(adoptee)
    {
    }

    void request() override
    {
        m_adoptee->specificRequest();
    }

private:
    Adoptee *m_adoptee;
};

// 适配了咖啡机和榨汁机的饮料机，采用对象适配器实现
class Beverage
{
public:
    virtual void getBeverage() = 0;
};

class CoffeeMaker
{
public:
    CoffeeMaker() = default;

    void Brew()
    {
        std::cout << "Brewing coffee\n";
    }
};

class JuiceMaker
{
public:
    JuiceMaker() = default;

    void Squeeze()
    {
        std::cout << "Squeezing Juice\n";
    }
};

class AdapterObject : public Beverage
{
private:
    std::function<void(void)> m_request;

public:
    AdapterObject(CoffeeMaker *cm)
    {
        m_request = [cm]()
        {
            cm->Brew();
        };
    }

    AdapterObject(JuiceMaker *jm)
    {
        m_request = [jm]()
        {
            jm->Squeeze();
        };
    }

    // 对外公共接口
    void getBeverage() override
    {
        m_request();
    }
};

// 类适配器与对象适配器代码对比
// 对象适配器
class ObjectAdapter : public Target
{
public:
    // 源接口的实例化
    ObjectAdapter(Adoptee *adoptee)
        : m_adoptee(adoptee)
    {
    }

    void request() override
    {
        std::cout << "From ObjectAdapter: ";
        m_adoptee->specificRequest();
    }

private:
    Adoptee *m_adoptee;
};

// 类适配器
// 钻石继承
class ClassAdapter
    : public Target
    , private Adoptee
{
public:
    void request() override
    {
        std::cout << "From ClassAdapter: ";
        specificRequest();
    }
};

// ------------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    Adoptee *adoptee = new Adoptee();

    Target *target = new Adapter(adoptee);

    target->request();

    std::cout << "================================\n";
    CoffeeMaker  *CM = new CoffeeMaker();
    AdapterObject coffee(CM);
    coffee.getBeverage();

    JuiceMaker   *JM = new JuiceMaker();
    AdapterObject juice(JM);
    juice.getBeverage();

    std::cout << "================================\n";
    Adoptee       *adoptee_   = new Adoptee();
    ObjectAdapter *adapter_1 = new ObjectAdapter(adoptee_);
    ClassAdapter  *adapter_2 = new ClassAdapter();

    adapter_1->request();
    adapter_2->request();

    return 0;
}
