#include "10_10_2_person.hpp"

#include <string.h>

#include <iostream>


Person::Person(const std::string &ln, const char *fn)
{
    this->lname = ln;
    // this->fname = fn;
    strcpy_s(this->fname, strlen(fn) + 1, fn);
}

void Person::Show() const
{
    std::cout << fname << " " << lname << std::endl;
}

void Person::FormalShow() const
{
    std::cout << lname << " " << fname << std::endl;
}