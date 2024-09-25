#include "06_BigInteger.h"

void BigInteger::cutLeadingZero()
{
    while (num.back() == 0 && num.size() != 1)
    {
        num.pop_back();
    }
}

void BigInteger::setLength()
{
    cutLeadingZero();
    int tmp = num.back();
    if (tmp == 0)
    {
        length = 1;
    }
    else
    {
        length = (num.size() - 1) * 8;
        while (tmp > 0)
        {
            ++length;
            tmp /= 10;
        }
    }
}

BigInteger::BigInteger(int n)
{
    *this = n;
}

BigInteger::BigInteger(long long n)
{
    *this = n;
}

BigInteger::BigInteger(const char *n)
{
    *this = n;
}

BigInteger::BigInteger(const BigInteger &n)
{
    *this = n;
}

const BigInteger &BigInteger::operator=(int n)
{
    *this = (long long)n;
    return *this;
}

const BigInteger &BigInteger::operator=(long long n)
{
    num.clear();
    if (n == 0)
    {
        num.push_back(0);
    }
    if (n >= 0)
    {
        sign = true;
    }
    else if (n == LONG_LONG_MIN)
    {
        *this      = "9223372036854775808";
        this->sign = false;
        return *this;
    }
    else if (n < 0)
    {
        sign = false;
        n    = -n;
    }
    while (n != 0)
    {
        num.push_back(n % BASE);
        n /= BASE;
    }
    setLength();
    return *this;
}

const BigInteger &BigInteger::operator=(const char *n)
{
    int len  = strlen(n);
    int tmp  = 0;
    int ten  = 1;
    int stop = 0;
    num.clear();
    sign = (n[0] != '-');
    if (!sign)
    {
        stop = 1;
    }
    for (int i = len; i > stop; --i)
    {
        tmp += (n[i - 1] - '0') * ten;
        ten *= 10;
        if ((len - i) % 8 == 7)
        {
            num.push_back(tmp);
            tmp = 0;
            ten = 1;
        }
    }
    if ((len - stop) % WIDTH != 0)
    {
        num.push_back(tmp);
    }
    setLength();
    return *this;
}

const BigInteger &BigInteger::operator=(const BigInteger &n)
{
    num    = n.num;
    sign   = n.sign;
    length = n.length;
    return *this;
}

size_t BigInteger::size() const
{
    return length;
}

BigInteger BigInteger::e(size_t n) const
{
    int        tmp = n % 8;
    BigInteger ans;
    ans.length = n + 1;
    n /= 8;
    while (ans.num.size() <= n)
    {
        ans.num.push_back(0);
    }
    ans.num[n] = 1;
    while (tmp--)
    {
        ans.num[n] *= 10;
    }
    return ans * (*this);
}

BigInteger BigInteger::abs() const
{
    BigInteger ans(*this);
    ans.sign = true;
    return ans;
}

const BigInteger &BigInteger::operator+() const
{
    return *this;
}

BigInteger operator+(const BigInteger &a, const BigInteger &b)
{
    if (!b.sign)
    {
        return a - (-b);
    }
    if (!a.sign)
    {
        return b - (-a);
    }
    BigInteger ans;
    int        carry = 0;
    int        aa, bb;
    size_t     lena = a.num.size();
    size_t     lenb = b.num.size();
    size_t     len  = std::max(lena, lenb);
    ans.num.clear();
    for (size_t i = 0; i < len; ++i)
    {
        if (lena <= i)
        {
            aa = 0;
        }
        else
        {
            aa = a.num[i];
        }
        if (lenb <= i)
        {
            bb = 0;
        }
        else
        {
            bb = b.num[i];
        }
        ans.num.push_back((aa + bb + carry) % BigInteger::BASE);
        carry = (aa + bb + carry) / BigInteger::BASE;
    }
    if (carry > 0)
    {
        ans.num.push_back(carry);
    }
    ans.setLength();
    return ans;
}

const BigInteger &BigInteger::operator+=(const BigInteger &n)
{
    *this = *this + n;
    return *this;
}

const BigInteger &BigInteger::operator++()
{
    *this = *this + 1;
    return *this;
}

BigInteger BigInteger::operator++(int)
{
    BigInteger ans(*this);
    *this = *this + 1;
    return ans;
}

BigInteger BigInteger::operator-() const
{
    BigInteger ans(*this);
    if (ans != 0)
    {
        ans.sign = !ans.sign;
    }
    return ans;
}

BigInteger operator-(const BigInteger &a, const BigInteger &b)
{
    if (!b.sign)
    {
        return a + (-b);
    }
    if (!a.sign)
    {
        return -((-a) + b);
    }
    if (a < b)
    {
        return -(b - a);
    }
    BigInteger ans;
    int        carry = 0;
    int        aa, bb;
    size_t     lena = a.num.size();
    size_t     lenb = b.num.size();
    size_t     len  = std::max(lena, lenb);
    ans.num.clear();
    for (size_t i = 0; i < len; ++i)
    {
        aa = a.num[i];
        if (i >= lenb)
        {
            bb = 0;
        }
        else
        {
            bb = b.num[i];
        }
        ans.num.push_back((aa - bb - carry + BigInteger::BASE) % BigInteger::BASE);
        if (aa < bb + carry)
        {
            carry = 1;
        }
        else
        {
            carry = 0;
        }
    }
    ans.setLength();
    return ans;
}

const BigInteger &BigInteger::operator-=(const BigInteger &n)
{
    *this = *this - n;
    return *this;
}

const BigInteger &BigInteger::operator--()
{
    *this = *this - 1;
    return *this;
}

BigInteger BigInteger::operator--(int)
{
    BigInteger ans(*this);
    *this = *this - 1;
    return ans;
}

BigInteger operator*(const BigInteger &a, const BigInteger &b)
{
    size_t                 lena = a.num.size();
    size_t                 lenb = b.num.size();
    std::vector<long long> ansLL;
    for (size_t i = 0; i < lena; ++i)
    {
        for (size_t j = 0; j < lenb; ++j)
        {
            if (i + j >= ansLL.size())
            {
                ansLL.push_back((long long)a.num[i] * (long long)b.num[j]);
            }
            else
            {
                ansLL[i + j] += (long long)a.num[i] * (long long)b.num[j];
            }
        }
    }
    while (ansLL.back() == 0 && ansLL.size() != 1)
    {
        ansLL.pop_back();
    }
    size_t     len   = ansLL.size();
    long long  carry = 0;
    long long  tmp;
    BigInteger ans;
    ans.sign = (ansLL.size() == 1 && ansLL[0] == 0) || (a.sign == b.sign);
    ans.num.clear();
    for (size_t i = 0; i < len; ++i)
    {
        tmp = ansLL[i];
        ans.num.push_back((tmp + carry) % BigInteger::BASE);
        carry = (tmp + carry) / BigInteger::BASE;
    }
    if (carry > 0)
    {
        ans.num.push_back(carry);
    }
    ans.setLength();
    return ans;
}

const BigInteger &BigInteger::operator*=(const BigInteger &n)
{
    *this = *this * n;
    return *this;
}

BigInteger operator/(const BigInteger &a, const BigInteger &b)
{
    BigInteger aa(a.abs());
    BigInteger bb(b.abs());
    if (aa < bb)
    {
        return 0;
    }
    char *str = new char[aa.size() + 1];
    memset(str, 0, sizeof(char) * (aa.size() + 1));
    BigInteger tmp;
    int        lena = aa.length;
    int        lenb = bb.length;
    for (int i = 0; i <= lena - lenb; ++i)
    {
        tmp = bb.e(lena - lenb - i);
        while (aa >= tmp)
        {
            ++str[i];
            aa = aa - tmp;
        }
        str[i] += '0';
    }

    BigInteger ans(str);
    delete[] str;
    ans.sign = (ans == 0 || a.sign == b.sign);
    return ans;
}

const BigInteger &BigInteger::operator/=(const BigInteger &n)
{
    *this = *this / n;
    return *this;
}

BigInteger operator%(const BigInteger &a, const BigInteger &b)
{
    return a - a / b * b;
}

const BigInteger &BigInteger::operator%=(const BigInteger &n)
{
    *this = *this - *this / n * n;
    return *this;
}

bool operator<(const BigInteger &a, const BigInteger &b)
{
    if (a.sign && !b.sign)
    {
        return false;
    }
    else if (!a.sign && b.sign)
    {
        return true;
    }
    else if (a.sign && b.sign)
    {
        if (a.length < b.length)
        {
            return true;
        }
        else if (a.length > b.length)
        {
            return false;
        }
        else
        {
            size_t lena = a.num.size();
            for (int i = lena - 1; i >= 0; --i)
            {
                if (a.num[i] < b.num[i])
                {
                    return true;
                }
                else if (a.num[i] > b.num[i])
                {
                    return false;
                }
            }
            return false;
        }
    }
    else
    {
        return -b < -a;
    }
}

bool operator<=(const BigInteger &a, const BigInteger &b)
{
    return !(b < a);
}

bool operator>(const BigInteger &a, const BigInteger &b)
{
    return b < a;
}

bool operator>=(const BigInteger &a, const BigInteger &b)
{
    return !(a < b);
}

bool operator==(const BigInteger &a, const BigInteger &b)
{
    return !(a < b) && !(b < a);
}

bool operator!=(const BigInteger &a, const BigInteger &b)
{
    return (a < b) || (b < a);
}

bool operator||(const BigInteger &a, const BigInteger &b)
{
    return a != 0 || b != 0;
}

bool operator&&(const BigInteger &a, const BigInteger &b)
{
    return a != 0 && b != 0;
}

bool BigInteger::operator!()
{
    return *this == 0;
}

std::ostream &operator<<(std::ostream &out, const BigInteger &n)
{
    size_t len = n.num.size();
    if (!n.sign)
    {
        out << '-';
    }
    out << n.num.back();
    for (int i = len - 2; i >= 0; --i)
    {
        out << std::setw(BigInteger::WIDTH) << std::setfill('0') << n.num[i];
    }
    return out;
}

std::istream &operator>>(std::istream &in, BigInteger &n)
{
    std::string str;
    in >> str;
    size_t len = str.length();
    size_t i, start = 0;
    if (str[0] == '-')
    {
        start = 1;
    }
    if (str[start] == '\0')
    {
        return in;
    }
    for (i = start; i < len; ++i)
    {
        if (str[i] < '0' || str[i] > '9')
        {
            return in;
        }
    }
    n = str.c_str();
    return in;
}
