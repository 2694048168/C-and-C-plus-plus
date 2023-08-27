/**
 * @file 12_10_2_my_string.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __MY_STRING_HPP__
#define __MY_STRING_HPP__

#include <iostream>

class String
{
public:
    char *str; // pointer to string
private:
    int   len; // length of string

    static int       num_strings;  /* number of objects, 引用计数策略(delete memory) */
    static const int CIN_LIM = 80; // cin input limit

public:
    // constructors and other methods
    String(const char *s);  // constructor
    String();               // default constructor
    String(const String &); // copy constructor
    ~String();              // destructor

    int length() const
    {
        return len;
    }

    void set_len(unsigned length)
    {
        len = length;
    }

    void     string_upper();
    void     string_lower();
    unsigned count_char(char);

    // overloaded operator methods
    String     &operator=(const String &);
    String     &operator=(const char *);
    char       &operator[](int i);
    const char &operator[](int i) const;

    // overloaded operator friends
    friend bool operator<(const String &st, const String &st2);
    friend bool operator>(const String &st1, const String &st2);
    friend bool operator==(const String &st, const String &st2);

    friend std::ostream &operator<<(std::ostream &os, const String &st);
    friend std::istream &operator>>(std::istream &is, String &st);

    // overloaded =, + operators
    String        &operator+(const String &);
    String        &operator+(const char *);
    friend String &operator+(const char *str2, const String &str1);

    // static function
    static int HowMany();
};

#endif // !__MY_STRING_HPP__