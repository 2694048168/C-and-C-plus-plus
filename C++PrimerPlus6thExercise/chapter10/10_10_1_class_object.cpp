/**
 * @file 10_10_1_class_object.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

class BankAccount
{
public:
    BankAccount() {}

    // BankAccount(const std::string &name, const std::string &user, const double &account)
    // {
    //     this->m_name    = name;
    //     this->m_user    = user;
    //     this->m_account = account;
    // }

    BankAccount(const std::string &name, const std::string &user, const double &account)
        : m_name(name)
        , m_user(user)
        , m_account(account)
    {
    }

    BankAccount(BankAccount &&)                 = default;
    BankAccount(const BankAccount &)            = default;
    BankAccount &operator=(BankAccount &&)      = default;
    BankAccount &operator=(const BankAccount &) = default;

    void show_bank_account()
    {
        std::cout << "------------------------------\n";
        std::cout << "The Name: " << m_name << "\n";
        std::cout << "The User: " << m_user << "\n";
        std::cout << "The Account: " << m_account << "\n";
        std::cout << "------------------------------\n";
    }

    bool set_account(const double &money)
    {
        if (!money)
        {
            return false;
        }

        m_account += money;
        return true;
    }

    bool modify_account(const double &money)
    {
        if (m_account < money)
        {
            return false;
        }

        m_account -= money;
        return true;
    }

    ~BankAccount() {}

private:
    std::string m_name{"name"};
    std::string m_user{"user"};
    double      m_account{0.0};
};

/**
 * @brief 编写C++程序, 实现 class 的声明和实现定义, 并简单演示使用方式
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, const char **argv)
{
    BankAccount user1;
    user1.show_bank_account();

    if (!user1.set_account(66.6))
    {
        std::cout << "Your must be add money, is not empty!\n";
    }
    user1.show_bank_account();

    if (!user1.modify_account(36))
    {
        std::cout << "Your account is NOT such more money!\n";
    }
    user1.show_bank_account();

    // -------------------------
    BankAccount user2{"wei li", "820820", 120990.82};
    user2.show_bank_account();

    if (!user2.set_account(9.18))
    {
        std::cout << "Your must be add money, is not empty!\n";
    }
    user2.show_bank_account();

    // if (user2.modify_account(2000))
    if (!user2.modify_account(200000))
    {
        std::cout << "Your account is NOT such more money!\n";
    }
    user2.show_bank_account();

    return 0;
}