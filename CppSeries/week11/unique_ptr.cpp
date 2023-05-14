/**
 * @file unique_ptr.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @date 2023-05-14
 * @version 0.1.1
 *
 * @copyright Copyright (c) 2023
 *
 * @brief smart pointer in C++11 
 * @attention 
 *
 */

#include <iostream>
#include <memory>
#include <string>

class MyTime
{
    int hours;
    int minutes;

public:
    MyTime() : hours(0), minutes(0)
    {
        std::cout << "Constructor MyTime()" << std::endl;
    }
    MyTime(int m) : hours(0), minutes(m)
    {
        std::cout << "Constructor MyTime(int)" << std::endl;
        this->hours += this->minutes / 60;
        this->minutes %= 60;
    }
    MyTime(int h, int m) : hours(h), minutes(m)
    {
        std::cout << "Constructor MyTime(int,int)" << std::endl;
        this->hours += this->minutes / 60;
        this->minutes %= 60;
    }
    ~MyTime()
    {
        std::cout << "Destructor MyTime(). Bye!" << std::endl;
    }

    MyTime operator+(int m) const
    {
        MyTime sum;
        sum.minutes = this->minutes + m;
        sum.hours = this->hours;
        sum.hours += sum.minutes / 60;
        sum.minutes %= 60;
        return sum;
    }

    friend std::ostream &operator<<(std::ostream &os, const MyTime &t)
    {
        std::string str = std::to_string(t.hours) + " hours and " + std::to_string(t.minutes) + " minutes.";
        os << str;
        return os;
    }
};

/**
 * @brief main function
 */
int main(int argc, const char **argv)
{
    std::unique_ptr<MyTime> mt1(new MyTime(10));
    std::unique_ptr<MyTime> mt2 = std::make_unique<MyTime>(80); // c++17

    std::cout << "mt1: " << *mt1 << std::endl;
    std::cout << "mt2: " << *mt2 << std::endl;

    // std::unique_ptr<MyTime> mt3 = mt1; // error
    std::unique_ptr<MyTime> mt3 = std::move(mt1);
    // std::shared_ptr<MyTime> mt3 = std::move(mt1);//okay

    // std::cout << "mt1: " <<*mt1 << std::endl;
    std::cout << "mt3: " << *mt3 << std::endl;

    return 0;
}

/** Build(compile and link) commands via command-line.
 *
 * $ clang++ unique_ptr.cpp
 * $ clang++ unique_ptr.cpp -std=c++17
 * $ ./a.exe # on Windows
 * $ ./a.out # on Linux or Mac
 *
 */