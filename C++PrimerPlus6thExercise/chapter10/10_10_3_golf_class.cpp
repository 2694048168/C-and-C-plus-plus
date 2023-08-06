/**
 * @file 10_10_3_golf_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>
#include <string>

class Golf
{
private:
    std::string fullname{"wei lil"};
    int         handicap{42};

public:
    Golf() = default;

    // constructor
    Golf(const char *name, int hc)
        : fullname(name)
        , handicap(hc){};

    // classname (const classname &obj) {} ----> copy constructor
    Golf(const Golf &g)
    {
        this->fullname = g.fullname;
        this->handicap = g.handicap;
    }

    void set_handicap(const int hc)
    {
        this->handicap = hc;
    }

    void show_golf() const
    {
        std::cout << "The fullname: " << this->fullname;
        std::cout << " and the handicap is " << this->handicap << "\n";
        std::cout << "----------------------------------------------\n";
    }
};

/**
 * @brief 编写C++程序, 以 class 方式封装数据和对应的操作
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Golf golf1;
    golf1.show_golf();

    // -------------
    Golf golf2{"jx", 66};
    golf2.show_golf();
    golf2.set_handicap(99);
    golf2.show_golf();

    // ---------------
    Golf golf3(golf1);
    golf3.show_golf();
    golf3.set_handicap(88);
    golf3.show_golf();

    return 0;
}