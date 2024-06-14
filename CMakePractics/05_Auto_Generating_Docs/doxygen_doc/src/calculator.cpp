/**
 * ______________________________________________________
 * @file calculator.cpp
 *
 * Calculator implementation
 *
 * SPDX-License-Identifier:	MIT
 * ______________________________________________________
 */

#include <calc/calculator.hpp>

namespace ProjectName { namespace Calc {

double calculator::sum(double augend, double addend)
{
    return last_result = augend + addend;
}

double calculator::sub(double minuend, double subtrahend)
{
    return last_result = minuend - subtrahend;
}

double calculator::mul(double multiplicand, double multiplier)
{
    return last_result = multiplicand * multiplier;
}

double calculator::div(double dividend, double divisor)
{
    return last_result = dividend / divisor;
}

}} // namespace ProjectName::Calc