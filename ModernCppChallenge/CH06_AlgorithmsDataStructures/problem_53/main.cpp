/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Average rating of movies
 * @version 0.1
 * @date 2024-01-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

/**
 * @brief Average rating of movies
 * 
 * Write a program that calculates and prints the average rating of a list of movies. 
 * Each movie has a list of ratings from 1 to 10 (where 1 is the lowest and 10 
 * is the highest rating). In order to compute the rating, you must remove 5% of 
 * the highest and lowest ratings before computing their average. 
 * The result must be displayed with a single decimal point.
 * 
 * The problem requires the computing of a movie rating using a truncated mean. 
 * This is a statistical measure of a central tendency where the mean is calculated
 * after discarding parts of a probability distribution or sample at the high and low ends. Typically, this is done by
 * removing an equal amount of points at the two ends. 
 * For this problem, you are required to remove 5% of 
 * both the highest and lowest user ratings.
 * 
 * A function that calculates a truncated mean for a given range should do the following:
 * 1. Sort the range so that elements are ordered (either ascending or descending)
 * 2. Remove the required percentage of elements at both ends
 * 3. Count the sum of all remaining elements
 * 4. Compute the average by dividing the sum to the remaining count of elements
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
struct movie
{
    int              id;
    std::string      title;
    std::vector<int> ratings;
};

double truncatedMean(std::vector<int> values, const double percentage)
{
    std::sort(std::begin(values), std::end(values));

    auto remove_count = static_cast<size_t>(values.size() * percentage + 0.5);

    values.erase(std::begin(values), std::begin(values) + remove_count);
    values.erase(std::end(values) - remove_count, std::end(values));

    auto total = std::accumulate(std::cbegin(values), std::cend(values), 0ull,
                                 [](const auto sum, const auto e) { return sum + e; });

    return static_cast<double>(total) / values.size();
}

void print_movie_ratings(const std::vector<movie> &movies)
{
    for (const auto &m : movies)
    {
        std::cout << m.title << " : " << std::fixed << std::setprecision(1) << truncatedMean(m.ratings, 0.05)
                  << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    std::vector<movie> movies{
        {101,   "The Matrix",     {10, 9, 10, 9, 9, 8, 7, 10, 5, 9, 9, 8}},
        {102,    "Gladiator", {10, 5, 7, 8, 9, 8, 9, 10, 10, 5, 9, 8, 10}},
        {103, "Interstellar",    {10, 10, 10, 9, 3, 8, 8, 9, 6, 4, 7, 10}}
    };

    print_movie_ratings(movies);

    return 0;
}
