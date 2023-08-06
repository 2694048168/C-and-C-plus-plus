#include "10_10_4_sale_class.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>

SALES::Sales::Sales(const std::vector<double> vec)
{
    this->sales = vec;

    this->average = std::accumulate(vec.cbegin(), vec.cend(), 0.0) / vec.size();
    this->max     = *std::max_element(vec.cbegin(), vec.cend());
    this->min     = *std::min_element(vec.cbegin(), vec.cend());
}

SALES::Sales::Sales(const SALES::Sales &s)
{
    this->sales   = s.sales;
    this->average = s.average;
    this->max     = s.max;
    this->min     = s.min;
}

void SALES::Sales::append_value(const double val)
{
    this->sales.push_back(val);

    this->average = std::accumulate(this->sales.cbegin(), this->sales.cend(), 0.0) / this->sales.size();
    this->max     = *std::max_element(this->sales.cbegin(), this->sales.cend());
    this->min     = *std::min_element(this->sales.cbegin(), this->sales.cend());
}

void SALES::Sales::showSales() const
{
    std::cout << "-------------------------------------\n";
    std::cout << "The value of Sales: [ ";
    for (const auto &elem : this->sales)
    {
        std::cout << elem << " ";
    }
    std::cout << "]\n";

    std::cout << "The average of Sales: " << this->average << "\n";
    std::cout << "The max of Sales: " << this->max << "\n";
    std::cout << "The min of Sales: " << this->min << "\n";
    std::cout << "-------------------------------------\n";
}