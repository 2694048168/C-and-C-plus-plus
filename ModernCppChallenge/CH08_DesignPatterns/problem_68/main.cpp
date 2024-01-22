/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Generating random passwords
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <random>
#include <string>

/**
 * @brief Generating random passwords
 * 
 * Write a program that can generate random passwords according to some predefined rules.
 * Every password must have a configurable minimum length. In addition, 
 * it should be possible to include in the generation rules such as 
 * the presence of at least one digit, symbol, lower or uppercase character, and so on.
 * These additional rules must be configurable and composable.
 *
 */

/**
 * @brief Solution:
------------------------------------------------------ */
class PasswordGenerator
{
public:
    virtual std::string generate()                                        = 0;
    virtual std::string allowed_chars() const                             = 0;
    virtual size_t      length() const                                    = 0;
    virtual void        add(std::unique_ptr<PasswordGenerator> generator) = 0;

    virtual ~PasswordGenerator() {}
};

class BasicPasswordGenerator : public PasswordGenerator
{
    size_t len;

public:
    explicit BasicPasswordGenerator(const size_t len) noexcept
        : len(len)
    {
    }

    virtual std::string generate() override
    {
        throw std::runtime_error("not implemented");
    }

    virtual void add(std::unique_ptr<PasswordGenerator>) override
    {
        throw std::runtime_error("not implemented");
    }

    virtual size_t length() const noexcept override final
    {
        return len;
    }
};

class DigitGenerator : public BasicPasswordGenerator
{
public:
    explicit DigitGenerator(const size_t len) noexcept
        : BasicPasswordGenerator(len)
    {
    }

    virtual std::string allowed_chars() const override
    {
        return "0123456789";
    }
};

class SymbolGenerator : public BasicPasswordGenerator
{
public:
    explicit SymbolGenerator(const size_t len) noexcept
        : BasicPasswordGenerator(len)
    {
    }

    virtual std::string allowed_chars() const override
    {
        return "!@#$%^&*(){}[]?<>";
    }
};

class UpperLetterGenerator : public BasicPasswordGenerator
{
public:
    explicit UpperLetterGenerator(const size_t len) noexcept
        : BasicPasswordGenerator(len)
    {
    }

    virtual std::string allowed_chars() const override
    {
        return "ABCDEFGHIJKLMNOPQRSTUVXYWZ";
    }
};

class LowerLetterGenerator : public BasicPasswordGenerator
{
public:
    explicit LowerLetterGenerator(const size_t len) noexcept
        : BasicPasswordGenerator(len)
    {
    }

    virtual std::string allowed_chars() const override
    {
        return "abcdefghijklmnopqrstuvxywz";
    }
};

class CompositePasswordGenerator : public PasswordGenerator
{
    virtual std::string allowed_chars() const override
    {
        throw std::runtime_error("not implemented");
    };

    virtual size_t length() const override
    {
        throw std::runtime_error("not implemented");
    };

public:
    CompositePasswordGenerator()
    {
        auto seed_data = std::array<int, std::mt19937::state_size>{};
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        eng.seed(seq);
    }

    virtual std::string generate() override
    {
        std::string password;
        for (auto &generator : generators)
        {
            std::string                     chars = generator->allowed_chars();
            std::uniform_int_distribution<> ud(0, static_cast<int>(chars.length() - 1));

            for (size_t i = 0; i < generator->length(); ++i) password += chars[ud(eng)];
        }

        std::shuffle(std::begin(password), std::end(password), eng);

        return password;
    }

    virtual void add(std::unique_ptr<PasswordGenerator> generator) override
    {
        generators.push_back(std::move(generator));
    }

private:
    std::random_device                              rd;
    std::mt19937                                    eng;
    std::vector<std::unique_ptr<PasswordGenerator>> generators;
};

// ------------------------------
int main(int argc, char **argv)
{
    CompositePasswordGenerator generator;
    generator.add(std::make_unique<SymbolGenerator>(2));
    generator.add(std::make_unique<DigitGenerator>(2));
    generator.add(std::make_unique<UpperLetterGenerator>(2));
    generator.add(std::make_unique<LowerLetterGenerator>(4));

    auto password = generator.generate();
    std::cout << password << std::endl;

    return 0;
}
