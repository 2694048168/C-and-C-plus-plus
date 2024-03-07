/**
 * @file 07_class.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 现代C++代码组织方式之类
 * @version 0.1
 * @date 2024-03-05
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief C++中类四要素: 构造函数 + 析构函数 
 *          + 需要封装的数据结构 + 操作数据的成员方法
 * 
 */
class ArithmeticOperation
{
public:
    ArithmeticOperation()
        : m_number1(0.f)
        , m_number2(0.f)
    {
    }

    ArithmeticOperation(const float val1, const float val2)
        : m_number1(val1)
        , m_number2(val2)
    {
    }

    ~ArithmeticOperation() = default;

    inline void printMessage(const std::string &message)
    {
        std::cout << message << std::endl;
    }

    inline float getAddResult()
    {
        return m_number1 + m_number1;
    }

    inline float getMulResult()
    {
        return m_number1 * m_number2;
    }

private:
    float m_number1;
    float m_number2;
};

// ===================================
int main(int argc, const char **argv)
{
    ArithmeticOperation arithmetic(3.12f, 2.1f);
    arithmetic.printMessage("======== two number add ========");
    float sum = arithmetic.getAddResult();
    std::cout << '\t' << sum << std::endl;

    arithmetic.printMessage("======== two number mul ========");
    float mul = arithmetic.getMulResult();
    std::cout << '\t' << mul << std::endl;

    return 0;
}

// ===================================
// compile and link via Clang or GCC
// clang++ .\07_class.cpp -std=c++23
// g++ .\07_class.cpp -std=c++23
