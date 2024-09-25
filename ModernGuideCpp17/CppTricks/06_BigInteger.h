#ifndef __BIG_INTEGER_H__
#define __BIG_INTEGER_H__

/**
 * @file 06_BigInteger.h
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-09-25
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

class BigInteger
{
private:
    static const int BASE  = 100000000;
    static const int WIDTH = 8;
    bool             sign;
    size_t           length;
    std::vector<int> num;

    void cutLeadingZero();
    void setLength();

public:
    static const long long _LONG_LONG_MIN_ = 1LL << 63;

    BigInteger(int n = 0);
    BigInteger(long long n);
    BigInteger(const char *n);
    BigInteger(const BigInteger &n);

    const BigInteger &operator=(int n);
    const BigInteger &operator=(long long n);
    const BigInteger &operator=(const char *n);
    const BigInteger &operator=(const BigInteger &n);

    size_t     size() const;
    BigInteger e(size_t n) const;
    BigInteger abs() const;

    const BigInteger &operator+() const;
    friend BigInteger operator+(const BigInteger &a, const BigInteger &b);
    const BigInteger &operator+=(const BigInteger &n);
    const BigInteger &operator++();
    BigInteger        operator++(int);

    BigInteger        operator-() const;
    friend BigInteger operator-(const BigInteger &a, const BigInteger &b);
    const BigInteger &operator-=(const BigInteger &n);
    const BigInteger &operator--();
    BigInteger        operator--(int);

    friend BigInteger operator*(const BigInteger &a, const BigInteger &b);
    const BigInteger &operator*=(const BigInteger &n);

    friend BigInteger operator/(const BigInteger &a, const BigInteger &b);
    const BigInteger &operator/=(const BigInteger &n);

    friend BigInteger operator%(const BigInteger &a, const BigInteger &b);
    const BigInteger &operator%=(const BigInteger &n);

    friend bool operator<(const BigInteger &a, const BigInteger &b);
    friend bool operator<=(const BigInteger &a, const BigInteger &b);
    friend bool operator>(const BigInteger &a, const BigInteger &b);
    friend bool operator>=(const BigInteger &a, const BigInteger &b);
    friend bool operator==(const BigInteger &a, const BigInteger &b);
    friend bool operator!=(const BigInteger &a, const BigInteger &b);

    friend bool operator||(const BigInteger &a, const BigInteger &b);
    friend bool operator&&(const BigInteger &a, const BigInteger &b);
    bool        operator!();

    friend std::ostream &operator<<(std::ostream &out, const BigInteger &n);
    friend std::istream &operator>>(std::istream &in, BigInteger &n);
};

#endif /* __BIG_INTEGER_H__ */