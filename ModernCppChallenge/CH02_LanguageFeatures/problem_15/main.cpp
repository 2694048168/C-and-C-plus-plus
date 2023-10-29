/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-10-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <array>
#include <iostream>
#include <sstream>
#include <string>

/* IPv4 data type
Write a class that represents an IPv4 address. 
Implement the functions required to be able to read and write 
such addresses from or to the console. 
The user should be able to input values in dotted form, 
such as 127.0.0.1 or 168.192.0.100. 
This is also the form in which IPv4 addresses 
should be formatted to an output stream.
------------------------------------------------ */
class ipv4
{
public:
    constexpr ipv4()
        : data{{0}}
    {
    }

    constexpr ipv4(unsigned char a, unsigned char b, unsigned char c, unsigned char d)
        : data{
            {a, b, c, d}
    }
    {
    }

    explicit constexpr ipv4(unsigned long a)
        : data{
            {static_cast<unsigned char>((a >> 24) & 0xFF), static_cast<unsigned char>((a >> 16) & 0xFF),
             static_cast<unsigned char>((a >> 8) & 0xFF), static_cast<unsigned char>(a & 0xFF)}
    }
    {
    }

    ipv4(const ipv4 &other) noexcept
        : data(other.data)
    {
    }

    ipv4 &operator=(const ipv4 &other) noexcept
    {
        data = other.data;
        return *this;
    }

    std::string to_string() const
    {
        std::stringstream sstr;
        sstr << *this;
        return sstr.str();
    }

    constexpr unsigned long to_ulong() const noexcept
    {
        return (static_cast<unsigned long>(data[0]) << 24) | (static_cast<unsigned long>(data[1]) << 16)
             | (static_cast<unsigned long>(data[2]) << 8) | static_cast<unsigned long>(data[3]);
    }

    friend std::ostream &operator<<(std::ostream &os, const ipv4 &a)
    {
        os << static_cast<int>(a.data[0]) << '.' << static_cast<int>(a.data[1]) << '.' << static_cast<int>(a.data[2])
           << '.' << static_cast<int>(a.data[3]);
        return os;
    }

    friend std::istream &operator>>(std::istream &is, ipv4 &a)
    {
        char d1, d2, d3;
        int  b1, b2, b3, b4;
        is >> b1 >> d1 >> b2 >> d2 >> b3 >> d3 >> b4;
        if (d1 == '.' && d2 == '.' && d3 == '.')
            a = ipv4(b1, b2, b3, b4);
        else
            is.setstate(std::ios_base::failbit);

        return is;
    }

private:
    std::array<unsigned char, 4> data;
};

// -------------------------
int main(int argc, char **)
{
    ipv4 a(168, 192, 0, 1);
    std::cout << a << std::endl;
    std::cout << a.to_string() << std::endl;

    ipv4 b = a;
    ipv4 c;
    c = b;

    ipv4 ip;
    std::cout << ip << std::endl;
    std::cin >> ip;
    if (!std::cin.fail())
        std::cout << ip << std::endl;

    return 0;
}