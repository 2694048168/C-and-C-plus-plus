/**
 * @file 8_8_7_template.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-08-05
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <iostream>

template<typename T> // template A
void ShowArray(T arr[], int n);

template<typename T> // template B
void ShowArray(T *arr[], int n);

template<typename T> // template A
void SumArray(T arr[], int n);

template<typename T> // template B
void SumArray(T *arr[], int n);

struct debts
{
    char   name[50];
    double amount;
};

int main(int argc, const char **argv)
{
    int things[6] = {13, 31, 103, 301, 310, 130};

    struct debts mr_E[3] = {
        {"Ima Wolfe", 2400.0},
        { "Ura Foxe", 1300.0},
        {"Iby Stout", 1800.0}
    };
    double *pd[3];

    // set pointers to the amount members of the structures in mr_E
    for (int i = 0; i < 3; i++) pd[i] = &mr_E[i].amount;

    std::cout << "Listing Mr. E's counts of things:\n";
    // things is an array of int
    ShowArray(things, 6); // uses template A
    std::cout << "Listing Mr. E's debts:\n";
    // pd is an array of pointers to double
    ShowArray(pd, 3); // uses template B (more specialized)

    std::cout << "=======================================\n";

    std::cout << "the Sum of Mr. E's counts of things:\n";
    SumArray(things, 6);

    std::cout << "the Sum of Mr. E's debts:\n";
    SumArray(pd, 3);

    return 0;
}

template<typename T>
void ShowArray(T arr[], int n)
{
    std::cout << "template A\n";
    for (int i = 0; i < n; i++) std::cout << arr[i] << ' ';
    std::cout << std::endl;
}

template<typename T>
void ShowArray(T *arr[], int n)
{
    std::cout << "template B\n";
    for (int i = 0; i < n; i++) std::cout << *arr[i] << ' ';
    std::cout << std::endl;
}

template<typename T>
void SumArray(T arr[], int n)
{
    std::cout << "template A\n";
    static T sum;
    for (int i = 0; i < n; i++)
    {
        sum += arr[i];
    }
    std::cout << sum << std::endl;
}

template<typename T>
void SumArray(T *arr[], int n)
{
    std::cout << "template B\n";
    static T sum;
    for (int i = 0; i < n; i++)
    {
        sum += *arr[i];
    }
    std::cout << sum << std::endl;
}