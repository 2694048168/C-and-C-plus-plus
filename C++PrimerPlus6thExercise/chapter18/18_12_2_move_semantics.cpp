/**
 * @file 18_12_2_move_semantics.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-09-14
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <cstddef>
#include <iostream>
#include <string>

// TODO 还需要考虑健壮性, 多几个判断情况,
// 拷贝构造函数和拷贝赋值函数
// 移动构造函数和移动赋值函数
class Cpmv
{
public:
    struct Info
    {
        std::string qcode;
        std::string zcode;
    };

private:
    Info *pi;

public:
    Cpmv()
    {
        std::cout << "default constructor call\n";

        pi = nullptr;
    }

    Cpmv(const std::string &q, const std::string &z)
    {
        std::cout << "custom constructor call\n";

        pi->qcode = q;
        pi->zcode = z;
    }

    Cpmv(const Cpmv &cp)
    {
        std::cout << "copy constructor call\n";

        this->pi->qcode = cp.pi->qcode;
        this->pi->zcode = cp.pi->zcode;
    }

    Cpmv(Cpmv &&mv)
    {
        std::cout << "move constructor call\n";

        this->pi->qcode = mv.pi->qcode;
        this->pi->zcode = mv.pi->zcode;

        // mv 所拥有的对象所有权被移动了,
        mv.pi = nullptr;
    }

    ~Cpmv()
    {
        std::cout << "default deconstructor call\n";
        pi = nullptr;
    }

    Cpmv &operator=(const Cpmv &cp)
    {
        std::cout << "move constructor call\n";

        this->pi->qcode = cp.pi->qcode;
        this->pi->zcode = cp.pi->zcode;

        return *this;
    }

    Cpmv &operator=(Cpmv &&mv);

    Cpmv operator+(const Cpmv &obj) const
    {
        if (this == &obj)
        {
            return *this;
        }

        Cpmv temp;
        temp.pi->qcode = this->pi->qcode + obj.pi->qcode;
        temp.pi->zcode = this->pi->zcode + obj.pi->zcode;

        return temp;
    }

    void Display() const
    {
        if (pi != nullptr)
        {
            std::cout << "the qcode: " << pi->qcode << "\n";
            std::cout << "the zcode: " << pi->zcode << "\n";
        }
        else
        {
            std::cout << "the qcode: NULL\n";
            std::cout << "the zcode: NULL\n";
        }
    }
};

/**
 * @brief 编写C++程序, 利用 C++11 提供的移动语义
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    Cpmv obj1;
    obj1.Display();

    std::string q_str = "wei";
    std::string z_str = " li";

    Cpmv obj2(q_str, z_str);
    obj2.Display();

    Cpmv obj3 = obj1 + obj2;
    obj3.Display();

    return 0;
}