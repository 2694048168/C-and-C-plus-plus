/**
 * @file 9_6_1_golf.hpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef __GOLF_HPP__
#define __GOLF_HPP__

const unsigned int LEN = 40;

struct Golf
{
    char fullname[LEN];
    int  handicap;
};

// non-interactive version:
// function sets golf structure to provided name, handicap
// using values passed as arguments to the function
void set_golf(Golf &g, const char *name, int hc);

// interactive version:
// function solicits name and handicap from user
// and sets the members of g to the values entered
// returns 1 if name is entered, 0 if name is empty string
int set_golf(Golf &g);

// function resets handicap to new value
void handicap(Golf &g, const int hc);

// function displays contents of golf structure
void show_golf(const Golf &g);

#endif // !__GOLF_HPP__