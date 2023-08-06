/**
 * @file 10_10_7_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <string.h>

#include <iostream>

const unsigned SIZE = 20;

class BetelPro
{
private:
    char name[SIZE] = "Peloria";
    int  ci         = 0;

public:
    BetelPro() = default;

    BetelPro(const char *name, int ci = 50)
    {
        strcpy_s(this->name, strlen(name) + 1, name);
    }

    void set_ci(const int val)
    {
        this->ci = val;
    }

    void show_info() const
    {
        std::cout << "The full Name " << this->name;
        std::cout << " and the CI is " << this->ci << "\n";
        std::cout << "------------------------------------\n";
    }
};

/**
 * @brief 编写C++程序, 根据数据data 和操作 operators 的需求设计类 class
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    BetelPro pro1;
    pro1.show_info();
    pro1.set_ci(42);
    pro1.show_info();

    BetelPro pro2("wei li");
    pro2.show_info();
    pro2.set_ci(66);
    pro2.show_info();

    return 0;
}