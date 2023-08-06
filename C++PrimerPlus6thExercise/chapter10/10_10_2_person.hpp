/**
 * @file 10_10_2_person.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-06
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __PERSON_HPP__
#define __PERSON_HPP__

#include <string>

class Person
{
private:
    static const unsigned LIMIT = 25;
    std::string           lname;        // Person’s last name
    char                  fname[LIMIT]; // Person’s first name

public:
    Person()
    {
        lname    = "";
        fname[0] = '\0';
    }

    Person(const std::string &ln, const char *fn = "Hey");

    // the following methods display lname and fname
    void Show() const;       // first name last name format
    void FormalShow() const; // last name, first name format
};

#endif // !__PERSON_HPP__