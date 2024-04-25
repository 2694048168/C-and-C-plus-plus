/**
 * @file 04_const.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-04-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstdio>

/**
 * @brief ===== const正确性
 * 关键字 const (constant)的意思大概可以表述为"保证不修改", 是一种安全机制,
 * 可以防止成员变量被意外修改(以及带来潜在的破坏),
 * 可以在函数和类的定义中使用 const 来指定变量(通常是引用或指针), 表示该变量不会被该函数或类修改;
 * 如果代码试图修改 const 变量, 编译器会发出一个错误.
 * 
 * 1. const参数
 * 将参数标记为 const 可以防止在函数的作用域内修改它,
 *  const 指针或引用可以提供有效的机制来将对象传递到函数中供只读使用.
 * 
 * 2. const方法
 * 将方法标记为 const 表示承诺不会在 const 方法中修改当前对象的状态, 这些方法都是只读方法.
 * 要将方法标记为 const, 需要将 const 关键字放在参数列表之后, 但在方法体之前.
 * *const 引用和指针的持有者不能调用非 const 方法, 因为非 const 方法可能会修改对象的状态.
 * 
 * 3. const成员变量
 * 在成员的类型前添加关键字 const 即可标记 const 成员变量, const 成员变量在初始化后不能被修改.
 * 如果存在一个 const 引用指向该对象, 那它也不能被修改.
 * 有时想将成员变量标记为 const, 但也想用传递到构造函数中的参数来初始化该成员, 可以使用成员初始化列表.
 * 
 */
void printMessage(const char *message)
{
    printf("[INFO] %s", message);
    // message[0] = 'A'; // !ERROR
}

class testClass
{
public:
    testClass()  = default;
    ~testClass() = default;

    int getYear() const
    {
        return m_year;
    }

    void setYear(const int year)
    {
        this->m_year = year;
    }

private:
    int m_year{};
};

struct ClockOfTheLongNow
{
    ClockOfTheLongNow(int year)
        : year(year)
    {
    }

    int get_year() const
    {
        return year;
    }

    void add_year()
    {
        ++year;
    }

private:
    int year;
};

// const 引用和指针的持有者不能调用非 const 方法, 因为非 const 方法可能会修改对象的状态.
// 如果 get_year 没有被标记为 const 方法, 编译error
// 因为 clock 是一个const 引用, 不允许在 is_leap_year 中被修改.
bool is_leap_year(const ClockOfTheLongNow &clock)
{
    if (clock.get_year() % 4 > 0)
        return false;
    if (clock.get_year() % 100 > 0)
        return true;
    if (clock.get_year() % 400 > 0)
        return false;
    return true;
}

struct Avout
{
    const char *name = "Erasmas";

    ClockOfTheLongNow aper;
};

void does_not_compile(const Avout &avout)
{
    // avout.aper.add_year(); // !Compiler error: avout is const
}

// -----------------------------------
int main(int argc, const char **argv)
{
    printMessage("const for function params\n");

    testClass testObj;
    testObj.setYear(2025);
    printf("the year is: %d\n", testObj.getYear());

    auto clock = ClockOfTheLongNow{2024};
    printf("2024 is leap year: %s\n", is_leap_year(clock) ? "yes" : "no");

    return 0;
}
