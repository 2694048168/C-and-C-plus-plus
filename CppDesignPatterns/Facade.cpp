/**
 * @file Facade.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-24
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Facade.cpp -std=c++20
 * clang++ Facade.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>
#include <vector>

class SubSystem
{
public:
    virtual void operation() = 0;
    virtual ~SubSystem()     = default;
};

class SubSystem_A : public SubSystem
{
public:
    void operation_A()
    {
        std::cout << "Exec operation_A from SubSystem_A\n";
    }

    void operation() override
    {
        operation_A();
    }
};

class SubSystem_B : public SubSystem
{
public:
    void operation_B()
    {
        std::cout << "Exec operation_B from SubSystem_B\n";
    }

    void operation() override
    {
        operation_B();
    }
};

class SubSystem_C : public SubSystem
{
public:
    void operation_C()
    {
        std::cout << "Exec operation_C from SubSystem_C\n";
    }

    void operation() override
    {
        operation_C();
    }
};

class Facade
{
private:
    std::vector<SubSystem *> subsystems;

public:
    Facade()
    {
        subsystems.push_back(new SubSystem_A);
        subsystems.push_back(new SubSystem_B);
        subsystems.push_back(new SubSystem_C);
    }

    ~Facade()
    {
        for (auto *subsystem : subsystems)
        {
            delete subsystem;
        }
    }

    void executeOperations()
    {
        for (auto &subsystem : subsystems)
        {
            subsystem->operation();
        }
    }
};

// Demo1: 模拟计算机的集成
//subSystem
class Monitor
{
public:
    void turnOn()
    {
        std::cout << "Monitor turned on.\n";
    }

    void turnOff()
    {
        std::cout << "Monitor turned off.\n";
    }
};

//subSystem
class Keyboard
{
public:
    void pressKey(int keyCode)
    {
        std::cout << "Pressed key: " << keyCode << ".\n";
    }
};

//subSystem
class CPU
{
public:
    void start()
    {
        std::cout << "CPU started.\n";
    }

    void stop()
    {
        std::cout << "CPU stopped.\n";
    }
};

//Facade
class Computer
{
private:
    Monitor  monitor;
    Keyboard keyboard;
    CPU      cpu;

public:
    Computer() {}

    void turnOnAndStart()
    {
        monitor.turnOn();
        keyboard.pressKey(13);
        cpu.start();
    }

    void shutDown()
    {
        cpu.stop();
        monitor.turnOff();
    }
};

// Demo2: 模拟汽车的集成
// Subsystem 1
class Engine
{
public:
    void Start()
    {
        std::cout << "Engine started\n";
    }

    void Stop()
    {
        std::cout << "Engine stopped\n";
    }
};

// Subsystem 2
class Lights
{
public:
    void TurnOn()
    {
        std::cout << "Lights on\n";
    }

    void TurnOff()
    {
        std::cout << "Lights off\n";
    }
};

// Facade
class Car
{
private:
    Engine engine;
    Lights lights;

public:
    void StartCar()
    {
        engine.Start();
        lights.TurnOn();
        std::cout << "Car is ready to drive\n";
    }

    void StopCar()
    {
        lights.TurnOff();
        engine.Stop();
        std::cout << "Car has stopped\n";
    }
};

// ------------------------------------
int main(int argc, const char **argv)
{
    std::cout << "------------------------------------\n";
    Facade facade;
    facade.executeOperations();

    std::cout << "------------------------------------\n";
    Computer myComputer;
    myComputer.turnOnAndStart();
    myComputer.shutDown();

    std::cout << "------------------------------------\n";
    Car car;
    car.StartCar();
    car.StopCar();

    return 0;
}
