/**
 * @file ObserversClockTimer.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-12-26
 * 
 * @copyright Copyright (c) 2024
 * 
 * g++ ObserversClockTimer.cpp -std=c++20
 * clang++ ObserversClockTimer.cpp -std=c++20
 * 
 */

#include <iostream>
#include <vector>

// Demo1: 基于观察者模式实现的模拟时钟定时
class Subject;

class Observer
{
public:
    virtual ~Observer()            = default;
    virtual void Update(Subject &) = 0;
};

class Subject
{
public:
    virtual ~Subject() = default;

    void Attach(Observer &o)
    {
        observers.push_back(&o);
    }

    void Detach(Observer &o)
    {
        observers.erase(std::remove(observers.begin(), observers.end(), &o));
    }

    void Notify()
    {
        for (auto *o : observers)
        {
            o->Update(*this);
        }
    }

private:
    std::vector<Observer *> observers;
};

class ClockTimer : public Subject
{
public:
    void SetTime(int hour, int minute, int second)
    {
        this->hour   = hour;
        this->minute = minute;
        this->second = second;
        Notify();
    }

    int GetHour() const
    {
        return hour;
    }

    int GetMinute() const
    {
        return minute;
    }

    int GetSecond() const
    {
        return second;
    }

private:
    int hour;
    int minute;
    int second;
};

class DigitalClock : public Observer
{
public:
    explicit DigitalClock(ClockTimer &s)
        : subject(s)
    {
        subject.Attach(*this);
    }

    ~DigitalClock()
    {
        subject.Detach(*this);
    }

    void Update(Subject &theChangedSubject) override
    {
        if (&theChangedSubject == &subject)
        {
            Draw();
        }
    }

    void Draw()
    {
        int hour   = subject.GetHour();
        int minute = subject.GetMinute();
        int second = subject.GetSecond();
        std::cout << "Digital time is " << hour << ":" << minute << ":" << second << std::endl;
    }

private:
    ClockTimer &subject;
};

class AnalogClock : public Observer
{
public:
    explicit AnalogClock(ClockTimer &s)
        : subject(s)
    {
        subject.Attach(*this);
    }

    ~AnalogClock()
    {
        subject.Detach(*this);
    }

    void Update(Subject &theChangedSubject) override
    {
        if (&theChangedSubject == &subject)
        {
            Draw();
        }
    }

    void Draw()
    {
        int hour   = subject.GetHour();
        int minute = subject.GetMinute();
        int second = subject.GetSecond();
        std::cout << "Analog time is " << hour << ":" << minute << ":" << second << std::endl;
    }

private:
    ClockTimer &subject;
};

// -----------------------------------------------
int main(int /* argc */, const char ** /* argv */)
{
    std::cout << "------------------------------\n";
    ClockTimer   timer;
    DigitalClock digitalClock(timer);
    AnalogClock  analogClock(timer);
    timer.SetTime(14, 41, 36);
    timer.SetTime(18, 00, 00);

    std::cout << "------------------------------\n";

    return 0;
}
