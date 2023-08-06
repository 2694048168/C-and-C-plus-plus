#include "9_6_4_namespace.hpp"

#include <iostream>
#include <iterator>
#include <numeric>

void SALES::setSales(SALES::Sales &s, const double ar[], int n)
{
    for (size_t i = 0; i < n; ++i)
    {
        s.sales[i] = ar[i];
    }

    s.average = std::accumulate(ar, ar + n, 0.0) / n;
    s.max     = *std::max_element(ar, ar + n);
    s.min     = *std::min_element(ar, ar + n);
}

void SALES::setSales(SALES::Sales &s)
{
    double input;
    for (size_t i = 0; i < SALES::QUARTERS; ++i)
    {
        std::cout << "Please enter #" << (i + 1) << " value: ";
        std::cin >> input;
        s.sales[i] = input;
        std::cin.ignore();
    }

    s.average = std::accumulate(s.sales, s.sales + SALES::QUARTERS, 0.0) / SALES::QUARTERS;
    s.max     = *std::max_element(s.sales, s.sales + SALES::QUARTERS);
    s.min     = *std::min_element(s.sales, s.sales + SALES::QUARTERS);
}

void SALES::showSales(const SALES::Sales &s)
{
    std::cout << "-------------------------------------\n";
    std::cout << "The value of Sales: [ ";
    for (const auto &elem : s.sales)
    {
        std::cout << elem << " ";
    }
    std::cout << "]\n";

    std::cout << "The average of Sales: " << s.average << "\n";
    std::cout << "The max of Sales: " << s.max << "\n";
    std::cout << "The min of Sales: " << s.min << "\n";
    std::cout << "-------------------------------------\n";
}