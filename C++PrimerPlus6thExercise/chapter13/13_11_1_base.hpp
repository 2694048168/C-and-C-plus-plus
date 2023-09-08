/**
 * @file 13_11_1_base.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-27
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __BASE_HPP__
#define __BASE_HPP__

// base class
class Cd
{ // represents a CD disk
private:
    char  *performers;
    char  *label;
    // char   performers[50];
    // char   label[20];
    int    selections; // number of selections
    double playtime;   // playing time in minutes
public:
    Cd(const char *s1, const char *s2, int n, double x);
    Cd(const Cd &d);
    Cd();
    virtual ~Cd();

    virtual void Report() const; // reports all CD data
    Cd          &operator=(const Cd &d);
};

// Classic Class
class Classic : public Cd
{
private:
    // char pr_work[100];
    char *pr_work;

public:
    Classic(const char *sc, const char *s1, const char *s2, int n, double x);
    Classic(const Classic &d);
    Classic();
    ~Classic();

    virtual void Report() const;
    Classic     &operator=(const Classic &d);
};

#endif // !__BASE_HPP__
