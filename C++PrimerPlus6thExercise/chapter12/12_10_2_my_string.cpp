#include "12_10_2_my_string.hpp"

#include <string.h>
#include <vcruntime.h>

#include <cstring>

// initializing static class member
int String::num_strings = 0;

// static method
int String::HowMany()
{
    return num_strings;
}

// class methods
String::String(const char *s) // construct String from C string
{
    len = std::strlen(s);    // set size
    str = new char[len + 1]; // allot storage
    // std::strcpy(str, s);     // initialize pointer
    strcpy_s(str, strlen(s) + 1, s); // initialize pointer
    ++num_strings;                   // set object count
}

String::String() // default constructor
{
    len    = 4;
    str    = new char[1];
    str[0] = '\0'; // default string
    num_strings++;
}

String::String(const String &st)
{
    num_strings++;           // handle static member update
    len = st.len;            // same length
    str = new char[len + 1]; // allot space
    // std::strcpy(str, st.str); // copy string to new location
    strcpy_s(str, strlen(st.str) + 1, st.str); // copy string to new location
}

String::~String() // necessary destructor
{
    --num_strings; // required
    delete[] str;  // required
}

// overloaded operator methods
// assign a String to a String

String &String::operator=(const String &st)
{
    if (this == &st)
        return *this;
    delete[] str;
    len = st.len;
    str = new char[len + 1];
    // std::strcpy(str, st.str);
    strcpy_s(str, strlen(st.str) + 1, st.str);

    return *this;
}

// assign a C string to a String
String &String::operator=(const char *s)
{
    delete[] str;
    len = std::strlen(s);
    str = new char[len + 1];
    // std::strcpy(str, s);
    strcpy_s(str, strlen(s) + 1, s);

    return *this;
}

// read-write char access for non-const String
char &String::operator[](int i)
{
    return str[i];
}

// read-only char access for const String
const char &String::operator[](int i) const
{
    return str[i];
}

// overloaded operator friends
bool operator<(const String &st1, const String &st2)
{
    return (std::strcmp(st1.str, st2.str) < 0);
}

bool operator>(const String &st1, const String &st2)
{
    return st2 < st1;
}

bool operator==(const String &st1, const String &st2)
{
    return (std::strcmp(st1.str, st2.str) == 0);
}

// simple String output
std::ostream &operator<<(std::ostream &os, const String &st)
{
    os << st.str;
    return os;
}

// quick and dirty String input
std::istream &operator>>(std::istream &is, String &st)
{
    char temp[String::CIN_LIM];
    is.get(temp, String::CIN_LIM);
    if (is)
        st = temp;
    while (is && is.get() != '\n') continue;
    return is;
}

void String::string_upper()
{
    for (size_t i = 0; i < len; ++i)
    {
        if (islower(str[i]))
        {
            str[i] = toupper(str[i]);
        }
    }
}

void String::string_lower()
{
    for (size_t i = 0; i < len; ++i)
    {
        if (isupper(str[i]))
        {
            str[i] = tolower(str[i]);
        }
    }
}

unsigned String::count_char(char ch)
{
    static unsigned num_char = 0;

    for (size_t i = 0; i < len; ++i)
    {
        if (ch == str[i])
        {
            ++num_char;
        }
    }

    return num_char;
}

// overloaded =, + operators
String &String::operator+(const String &s)
{
    static String temp;
    temp.len = this->len + s.len;

    strcpy_s(temp.str, this->len, this->str);
    strcpy_s(temp.str + this->len, s.len, s.str);

    return temp;
}

String &String::operator+(const char *s)
{
    static String temp;
    temp.len = this->len + strlen(s) + 2;

    strcpy_s(temp.str, this->len + 1, this->str);
    strcpy_s(temp.str + this->len + 1, strlen(s) + 1, s);

    return temp;
}

String &operator+(const char *str2, const String &str1)
{
    static String temp;
    temp.set_len(str1.length() + 1 + strlen(str2) + 1);

    // TODO 权限控制，str public?
    strcpy_s(temp.str, str1.length() + 1, str1.str);
    strcpy_s(temp.str + str1.length() + 1, strlen(str2) + 1, str2);

    return temp;
}