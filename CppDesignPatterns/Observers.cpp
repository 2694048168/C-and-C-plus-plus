/**
 * @file Observers.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-26
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ Observers.cpp -std=c++20
 * clang++ Observers.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

// Demo1: subject只完成通知
class Observer
{
public:
    virtual void update() = 0;
    virtual ~Observer()   = default;
};

class ConcreteObserver : public Observer
{
public:
    ConcreteObserver(std::string name)
    {
        observer_name = name;
    }

    void update()
    {
        std::cout << observer_name << " received notify.\n";
    }

private:
    std::string observer_name = "";
};

class Subject
{
private:
    // 观察者集合
    std::vector<Observer *> observers;

public:
    // 添加观察者
    void attach(Observer *observer)
    {
        observers.push_back(observer);
    }

    // 移除观察者
    void detach(Observer *observer)
    {
        for (auto it = observers.begin(); it != observers.end(); ++it)
        {
            if (*it == observer)
            {
                observers.erase(it);
                break;
            }
        }
    }

    // 通知观察者
    void notify()
    {
        for (auto observer : observers)
        {
            observer->update();
        }
    }
};

// Demo2: subject完成通知并传参
class Observer_
{
public:
    virtual void update(int data) = 0;
};

class ConcreteObserver_ : public Observer_
{
public:
    ConcreteObserver_(std::string name)
    {
        observer_name = name;
    }

    void update(int data) override
    {
        std::cout << observer_name << " received data: " << data << '\n';
    }

private:
    std::string observer_name = "";
};

class Subject_
{
public:
    virtual void attach(Observer_ *observer) = 0;
    virtual void detach(Observer_ *observer) = 0;
    virtual void notify(int data)            = 0;
};

class ConcreteSubject : public Subject_
{
private:
    std::vector<Observer_ *> observers;

public:
    void attach(Observer_ *observer) override
    {
        observers.push_back(observer);
    }

    void detach(Observer_ *observer) override
    {
        for (auto it = observers.begin(); it != observers.end(); ++it)
        {
            if (*it == observer)
            {
                observers.erase(it);
                break;
            }
        }
    }

    void notify(int data) override
    {
        for (auto observer : observers)
        {
            observer->update(data);
        }
    }
};

// Demo1: 基于观察者模式实现的模拟时钟定时
// Demo2：基于观察者模式实现的模拟天气预报

// -----------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "------------------------------\n";
    Subject          subject;
    ConcreteObserver observer1("observer_1");
    ConcreteObserver observer2("observer_2");

    subject.attach(&observer1);
    subject.attach(&observer2);
    subject.notify();
    subject.detach(&observer2);
    subject.notify();

    std::cout << "------------------------------\n";
    ConcreteSubject   subject_;
    ConcreteObserver_ observer1_("observer_1_");
    ConcreteObserver_ observer2_("observer_2_");
    ConcreteObserver_ observer3_("observer_3_");
    subject_.attach(&observer1_);
    subject_.attach(&observer2_);
    subject_.attach(&observer3_);

    subject_.notify(30);
    subject_.detach(&observer1_);
    subject_.notify(40);
    subject_.detach(&observer2_);
    subject_.notify(90);

    std::cout << "------------------------------\n";

    return 0;
}
