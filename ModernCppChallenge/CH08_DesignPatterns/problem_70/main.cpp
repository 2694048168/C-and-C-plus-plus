/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Approval system
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <iostream>
#include <memory>
#include <string>
#include <string_view>

/**
 * @brief Approval system
 * 
 * Write a program for a purchasing department of a company that allows employees
 *  to approve new purchases (or expenses). However, based on their position, 
 * each employee may only approve expenses up to a predefined limit. 
 * For instance, regular employees can approve expenses up to 1,000 currency units,
 *  team managers up to 10,000, and the department manager up to 100,000.
 * Any expense greater than that must be explicitly approved by the company president.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
class role
{
public:
    virtual double approval_limit() const noexcept = 0;

    virtual ~role() {}
};

class employee_role : public role
{
public:
    virtual double approval_limit() const noexcept override
    {
        return 1000;
    }
};

class team_manager_role : public role
{
public:
    virtual double approval_limit() const noexcept override
    {
        return 10000;
    }
};

class department_manager_role : public role
{
public:
    virtual double approval_limit() const noexcept override
    {
        return 100000;
    }
};

class president_role : public role
{
public:
    virtual double approval_limit() const noexcept override
    {
        return std::numeric_limits<double>::max();
    }
};

struct expense
{
    double      amount;
    std::string description;

    expense(const double amount, std::string_view desc)
        : amount(amount)
        , description(desc)
    {
    }
};

class employee
{
public:
    explicit employee(std::string_view name, std::unique_ptr<role> ownrole)
        : name(name)
        , own_role(std::move(ownrole))
    {
    }

    void set_direct_manager(std::shared_ptr<employee> manager)
    {
        direct_manager = manager;
    }

    void approve(const expense &e)
    {
        if (e.amount <= own_role->approval_limit())
            std::cout << name << " approved expense '" << e.description << "', cost=" << e.amount << std::endl;

        else if (direct_manager != nullptr)
            direct_manager->approve(e);
    }

private:
    std::string               name;
    std::unique_ptr<role>     own_role;
    std::shared_ptr<employee> direct_manager;
};

// ------------------------------
int main(int argc, char **argv)
{
    auto john = std::make_shared<employee>("john smith", std::make_unique<employee_role>());

    auto robert = std::make_shared<employee>("robert booth", std::make_unique<team_manager_role>());

    auto david = std::make_shared<employee>("david jones", std::make_unique<department_manager_role>());

    auto cecil = std::make_shared<employee>("cecil williamson", std::make_unique<president_role>());

    john->set_direct_manager(robert);
    robert->set_direct_manager(david);
    david->set_direct_manager(cecil);

    john->approve(expense{500, "magazins"});
    john->approve(expense{5000, "hotel accomodation"});
    john->approve(expense{50000, "conference costs"});
    john->approve(expense{200000, "new lorry"});

    return 0;
}
