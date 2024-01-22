/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Generating social security numbers
 * @version 0.1
 * @date 2024-01-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <string_view>

/**
 * @brief Generating social security numbers
 * 
 * Write a program that can generate social security numbers for two countries, 
 * Northeria and Southeria, that have different but similar formats for the numbers:
 * 1.  In Northeria, the numbers have the format SYYYYMMDDNNNNNC, where S is a digit
 * representing the sex, 9 for females and 7 for males, YYYYMMDD is the birth date,
 * NNNNN is a five-digit random number, unique for a day (meaning that the same
 * number can appear twice for two different dates, but not the same date), 
 * and C is a digit picked so that the checksum computed as described later
 *  is a multiple of 11.
 * 2.  In Southeria, the numbers have the format SYYYYMMDDNNNNC, where S is a digit
 * representing the sex, 1 for females and 2 for males, YYYYMMDD is the birth date,
 * NNNN is a four-digit random number, unique for a day, and C is a digit picked 
 * so that the checksum computed as described below is a multiple of 10.
 *
 * The checksum in both cases is a sum of all the digits, each multiplied 
 * by its weight (the position from the most significant digit to the least).
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
enum class SexType
{
    female,
    male
};

class SocialNumberGenerator
{
protected:
    virtual int sex_digit(const SexType sex) const noexcept                                = 0;
    virtual int next_random(const unsigned year, const unsigned month, const unsigned day) = 0;
    virtual int modulo_value() const noexcept                                              = 0;

    SocialNumberGenerator(const int min, const int max)
        : ud(min, max)
    {
        std::random_device rd;
        auto               seed_data = std::array<int, std::mt19937::state_size>{};
        std::generate(std::begin(seed_data), std::end(seed_data), std::ref(rd));
        std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
        eng.seed(seq);
    }

public:
    std::string generate(const SexType sex, const unsigned year, const unsigned month, const unsigned day)
    {
        std::stringstream snumber;

        snumber << sex_digit(sex);

        snumber << year << month << day;

        snumber << next_random(year, month, day);

        auto number = snumber.str();

        auto index = number.length();
        auto sum   = std::accumulate(std::begin(number), std::end(number), 0u,
                                     [&index](const unsigned int s, const char c)
                                     { return s + static_cast<unsigned int>(index-- * (c - '0')); });

        auto rest = sum % modulo_value();
        snumber << modulo_value() - rest;

        return snumber.str();
    }

    virtual ~SocialNumberGenerator() {}

protected:
    std::map<unsigned, int>         cache;
    std::mt19937                    eng;
    std::uniform_int_distribution<> ud;
};

class SoutheriaSocialNumberGenerator final : public SocialNumberGenerator
{
public:
    SoutheriaSocialNumberGenerator()
        : SocialNumberGenerator(1000, 9999)
    {
    }

protected:
    virtual int sex_digit(const SexType sex) const noexcept override
    {
        if (sex == SexType::female)
            return 1;
        else
            return 2;
    }

    virtual int next_random(const unsigned year, const unsigned month, const unsigned day) override
    {
        auto key = year * 10000 + month * 100 + day;
        while (true)
        {
            auto number = ud(eng);
            auto pos    = cache.find(number);
            if (pos == std::end(cache))
            {
                cache[key] = number;
                return number;
            }
        }
    }

    virtual int modulo_value() const noexcept override
    {
        return 11;
    }
};

class NortheriaSocialNumberGenerator final : public SocialNumberGenerator
{
public:
    NortheriaSocialNumberGenerator()
        : SocialNumberGenerator(10000, 99999)
    {
    }

protected:
    virtual int sex_digit(const SexType sex) const noexcept override
    {
        if (sex == SexType::female)
            return 9;
        else
            return 7;
    }

    virtual int next_random(const unsigned year, const unsigned month, const unsigned day) override
    {
        auto key = year * 10000 + month * 100 + day;
        while (true)
        {
            auto number = ud(eng);
            auto pos    = cache.find(number);
            if (pos == std::end(cache))
            {
                cache[key] = number;
                return number;
            }
        }
    }

    virtual int modulo_value() const noexcept override
    {
        return 11;
    }
};

class SocialNumberGeneratorFactory
{
public:
    SocialNumberGeneratorFactory()
    {
        generators["northeria"] = std::make_unique<NortheriaSocialNumberGenerator>();
        generators["southeria"] = std::make_unique<SoutheriaSocialNumberGenerator>();
    }

    SocialNumberGenerator *get_generator(std::string_view country) const
    {
        auto it = generators.find(country.data());
        if (it != std::end(generators))
            return it->second.get();

        throw std::runtime_error("invalid country");
    }

private:
    std::map<std::string, std::unique_ptr<SocialNumberGenerator>> generators;
};

// ------------------------------
int main(int argc, char **argv)
{
    SocialNumberGeneratorFactory factory;

    auto sn1 = factory.get_generator("northeria")->generate(SexType::female, 2017, 12, 25);
    auto sn2 = factory.get_generator("northeria")->generate(SexType::female, 2017, 12, 25);
    auto sn3 = factory.get_generator("northeria")->generate(SexType::male, 2017, 12, 25);

    auto ss1 = factory.get_generator("southeria")->generate(SexType::female, 2017, 12, 25);
    auto ss2 = factory.get_generator("southeria")->generate(SexType::female, 2017, 12, 25);
    auto ss3 = factory.get_generator("southeria")->generate(SexType::male, 2017, 12, 25);

    return 0;
}
