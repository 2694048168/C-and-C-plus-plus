/**
 * @file 36_Delegate_Construct.cpp
 * @author Wei Li (Ithaca) (weili_yzzcq@163.com)
 * @brief 现代 C++ 特性：委托构造
 * @version 0.1
 * @date 2025-08-05
 * 
 * @copyright Copyright (c) 2025
 * 
 * g++ 36_Delegate_Construct.cpp -std=c++20
 * clang++ 36_Delegate_Construct.cpp -std=c++20
 * 
 */

#include <iostream>
#include <string>

/**
 * @brief delegate construct
 * 1. 复用构造逻辑
 * 2. 代码的高维护性
 * 3. 性能优势, 成员只被初始化一次
 * 
 */
class Account
{
private:
    std::string mID;
    std::string mName;
    double      mCoin;

public:
    Account(std::string id)
        : Account(id, "money") /* delegate construct */
    {
    }

    Account(std::string id, std::string name)
        : Account(id, name, 0.0) /* delegate construct */
    {
    }

    Account(std::string id, std::string name, double coin)
        : mID{id}
        , mName{name}
        , mCoin{coin}
    {
    }

    // Friend function to overload operator<<
    friend std::ostream &operator<<(std::ostream &os, const Account &obj)
    {
        os << obj.mName << " Account " << obj.mID << " with " << obj.mCoin << " $\n";
        return os;
    }
};

// -------------------------------------
int main(int argc, const char *argv[])
{
    Account account1{"820820"};
    std::cout << account1;

    Account account2{"820820", "wei li"};
    std::cout << account2;

    Account account3{"820820", "Wei Li", 2000.34};
    std::cout << account3;
    
    return 0;
}
