/**
 * @file FactoryPattern.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 抽象工厂模式
 * @version 0.1
 * @date 2024-01-15
 * 
 * @copyright Copyright (c) 2024
 * 
 */

/**
 * @brief   基础型  标准型  旗舰型 
 *    船体   木头    钢铁    合成金属
 *    动力   手动    内燃机  核能
 *    武器   枪      速射炮  激光
 * 
 */

#include <iostream>
#include <string>

// 船体
class ShipBody
{
public:
    virtual std::string getBody() = 0;

    // ! 利用到多态, 那么一定要实现一个抽象基类的析构函数
    virtual ~ShipBody() {}
};

class WoodBody : public ShipBody
{
public:
    std::string getBody() override
    {
        return "Using <Wood> as body of ship...";
    }

    ~WoodBody()
    {
        std::cout << "~WoodBody\n";
    }
};

class IronBody : public ShipBody
{
public:
    std::string getBody() override
    {
        return "Using <Iron> as body of ship...";
    }

    ~IronBody()
    {
        std::cout << "~IronBody\n";
    }
};

class MetalBody : public ShipBody
{
public:
    std::string getBody() override
    {
        return "Using <Metal> as body of ship...";
    }

    ~MetalBody()
    {
        std::cout << "~MetalBody\n";
    }
};

// 引擎
class Engine
{
public:
    virtual std::string getEngine() = 0;

    // ! 利用到多态, 那么一定要实现一个抽象基类的析构函数
    virtual ~Engine() {}
};

class Human : public Engine
{
public:
    std::string getEngine() override
    {
        return "Using <Human> as Engine of ship...";
    }

    ~Human()
    {
        std::cout << "~Human\n";
    }
};

class Diesel : public Engine
{
public:
    std::string getEngine() override
    {
        return "Using <Diesel> as Engine of ship...";
    }

    ~Diesel()
    {
        std::cout << "~Diesel\n";
    }
};

class Nuclear : public Engine
{
public:
    std::string getEngine() override
    {
        return "Using <Nuclear> as Engine of ship...";
    }

    ~Nuclear()
    {
        std::cout << "~Nuclear\n";
    }
};

// 武器
class Weapon
{
public:
    virtual std::string getWeapon() = 0;

    // ! 利用到多态, 那么一定要实现一个抽象基类的析构函数
    virtual ~Weapon() {}
};

class Gun : public Weapon
{
public:
    std::string getWeapon() override
    {
        return "Using <Gun> as Weapon of ship...";
    }

    ~Gun()
    {
        std::cout << "~Gun\n";
    }
};

class Cannon : public Weapon
{
public:
    std::string getWeapon() override
    {
        return "Using <Cannon> as Weapon of ship...";
    }

    ~Cannon()
    {
        std::cout << "~Cannon\n";
    }
};

class Laser : public Weapon
{
public:
    std::string getWeapon() override
    {
        return "Using <Laser> as Weapon of ship...";
    }

    ~Laser()
    {
        std::cout << "~Laser\n";
    }
};

// 组装成为船
class Ship
{
public:
    std::string getProperty()
    {
        return m_pBody->getBody() + m_pEngine->getEngine() + m_pWeapon->getWeapon();
    }

    Ship(ShipBody *body, Engine *engine, Weapon *weapon)
        : m_pBody(body)
        , m_pEngine(engine)
        , m_pWeapon(weapon)
    {
    }

    // 组合或者聚合关系, 涉及是否需要析构该对象
    ~Ship()
    {
        if (m_pBody != nullptr)
        {
            delete m_pBody;
            m_pBody = nullptr;
        }

        if (m_pEngine != nullptr)
        {
            delete m_pEngine;
            m_pEngine = nullptr;
        }

        if (m_pWeapon != nullptr)
        {
            delete m_pWeapon;
            m_pWeapon = nullptr;
        }
    }

private:
    ShipBody *m_pBody;
    Engine   *m_pEngine;
    Weapon   *m_pWeapon;
};

// 抽象工厂类
class AbstractFactory
{
public:
    virtual Ship *createShip() = 0;

    virtual ~AbstractFactory() {}
};

// 生产基础型的工厂类
class BasicShip : public AbstractFactory
{
public:
    Ship *createShip() override
    {
        Ship *ship = new Ship(new WoodBody, new Human, new Gun);
        std::cout << "The [Basic] Ship is Done successfully\n";
        return ship;
    }

    BasicShip()
    {
        std::cout << "~BasicShip\n";
    }
};

// 生产标准型的工厂类
class StandardShip : public AbstractFactory
{
public:
    Ship *createShip() override
    {
        Ship *ship = new Ship(new IronBody, new Diesel, new Cannon);
        std::cout << "The [Standard] Ship is Done successfully\n";
        return ship;
    }

    StandardShip()
    {
        std::cout << "~StandardShip\n";
    }
};

// 生产旗舰型的工厂类
class UltimateShip : public AbstractFactory
{
public:
    Ship *createShip() override
    {
        Ship *ship = new Ship(new MetalBody, new Nuclear, new Laser);
        std::cout << "The [Ultimate] Ship is Done successfully\n";
        return ship;
    }

    UltimateShip()
    {
        std::cout << "~UltimateShip\n";
    }
};

// ====================================
int main(int argc, const char **argv)
{
    AbstractFactory *pFactory = new UltimateShip;

    Ship *pShip = pFactory->createShip();
    std::cout << pShip->getProperty() << std::endl;

    if (pFactory != nullptr)
    {
        delete pFactory;
        pFactory = nullptr;
    }

    if (pShip != nullptr)
    {
        delete pShip;
        pShip = nullptr;
    }

    return 0;
}
