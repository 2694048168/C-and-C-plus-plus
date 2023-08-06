#include "9_6_1_golf.hpp"

#include <cstring>
#include <iostream>
#include <string>

void set_golf(Golf &g, const char *name, const int hc)
{
    // To copy the content into the array,
    // need to use strcpy() from string.h (char array) or memcpy() in general.
    // g.fullname = name;
    // strcpy_s(g.fullname, strlen(name) + 1, name);
    std::memcpy(g.fullname, name, strlen(name) + 1);

    g.handicap = hc;
}

int set_golf(Golf &g)
{
    std::cout << "Please enter your fullname: ";
    std::string name;
    std::getline(std::cin, name);
    std::memcpy(g.fullname, name.c_str(), strlen(name.c_str()) + 1);

    std::cout << "Please enter your handicap: ";
    std::cin >> g.handicap;
    std::cin.ignore();

    std::string str(g.fullname);
    if (str.empty())
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

void handicap(Golf &g, const int hc)
{
    g.handicap = hc;
}

void show_golf(const Golf &g)
{
    std::cout << "The fullname: " << g.fullname;
    std::cout << " and the handicap is " << g.handicap << std::endl;
}