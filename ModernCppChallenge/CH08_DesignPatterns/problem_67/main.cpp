/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Validating passwords
 * @version 0.1
 * @date 2024-01-09
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cassert>
#include <cctype>
#include <memory>
#include <string_view>
#include <utility>

/**
 * @brief Validating passwords
 * 
 * Write a program that validates password strength based on predefined rules, 
 * which may then be selected in various combinations. At a minimum, every password 
 * must meet a minimum length requirement. In addition, other rules could be enforced,
 * such as the presence of at least one symbol, digit, uppercase and lowercase letter,
 * and so on.
 * 
 * 编写一个基于预定义规则验证密码强度的程序, 该规则可能然后以各种组合进行选择.
 * 每个密码至少必须满足最小长度要求. 此外, 还可以执行其他规则,
 * 例如至少存在一个符号、数字、大小写字母等.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
class PasswordValidator
{
public:
    virtual bool validate(std::string_view password) = 0;

    // must virtual deconstructor function for polymorphic
    virtual ~PasswordValidator() {}
};

class LengthValidator final : public PasswordValidator
{
public:
    LengthValidator(unsigned int min_length)
        : length(min_length)
    {
    }

    virtual bool validate(std::string_view password) override
    {
        return password.length() >= length;
    }

private:
    unsigned int length;
};

class PasswordValidatorDecorator : public PasswordValidator
{
public:
    explicit PasswordValidatorDecorator(std::unique_ptr<PasswordValidator> validator)
        : inner(std::move(validator))
    {
    }

    virtual bool validate(std::string_view password) override
    {
        return inner->validate(password);
    }

private:
    std::unique_ptr<PasswordValidator> inner;
};

class DigitPasswordValidator final : public PasswordValidatorDecorator
{
public:
    explicit DigitPasswordValidator(std::unique_ptr<PasswordValidator> validator)
        : PasswordValidatorDecorator(std::move(validator))
    {
    }

    virtual bool validate(std::string_view password) override
    {
        if (!PasswordValidatorDecorator::validate(password))
            return false;

        return password.find_first_of("0123456789") != std::string::npos;
    }
};

class CasePasswordValidator final : public PasswordValidatorDecorator
{
public:
    explicit CasePasswordValidator(std::unique_ptr<PasswordValidator> validator)
        : PasswordValidatorDecorator(std::move(validator))
    {
    }

    virtual bool validate(std::string_view password) override
    {
        if (!PasswordValidatorDecorator::validate(password))
            return false;

        bool has_lower = false;
        bool has_upper = false;

        for (size_t i = 0; i < password.length() && !(has_upper && has_lower); ++i)
        {
            if (std::islower(password[i]))
                has_lower = true;
            else if (std::isupper(password[i]))
                has_upper = true;
        }
        return has_lower && has_upper;
    }
};

class SymbolPasswordValidator final : public PasswordValidatorDecorator
{
public:
    explicit SymbolPasswordValidator(std::unique_ptr<PasswordValidator> validator)
        : PasswordValidatorDecorator(std::move(validator))
    {
    }

    virtual bool validate(std::string_view password) override
    {
        if (!PasswordValidatorDecorator::validate(password))
            return false;

        return password.find_first_of("!@#$%^&*(){}[]?<>") != std::string::npos;
    }
};

// ------------------------------
int main(int argc, char **argv)
{
    {
        auto validator = std::make_unique<LengthValidator>(8);

        assert(validator->validate("abc123!@#"));
        assert(!validator->validate("abc123"));
    }

    {
        auto validator = std::make_unique<DigitPasswordValidator>(std::make_unique<LengthValidator>(8));

        assert(validator->validate("abc123!@#"));
        assert(!validator->validate("abcde!@#"));
    }

    {
        auto validator = std::make_unique<CasePasswordValidator>(
            std::make_unique<DigitPasswordValidator>(std::make_unique<LengthValidator>(8)));

        assert(validator->validate("Abc123!@#"));
        assert(!validator->validate("abc123!@#"));
    }

    {
        auto validator = std::make_unique<SymbolPasswordValidator>(std::make_unique<CasePasswordValidator>(
            std::make_unique<DigitPasswordValidator>(std::make_unique<LengthValidator>(8))));

        assert(validator->validate("Abc123!@#"));
        assert(!validator->validate("Abc123567"));
    }

    return 0;
}
