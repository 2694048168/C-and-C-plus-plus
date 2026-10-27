/**
 * @file Recipe.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include <string>
#include <vector>

namespace Ithaca {

struct NutritionInfo
{
    int calories{};
    int protein_grams{};
    int carbs_grams{};
    int fat_grams{};
};

struct Recipe
{
    std::string              title{};
    std::vector<std::string> ingredients{};
    std::vector<std::string> step{};
    NutritionInfo            nutrition{};

    int prep_time_minutes{};
    int cook_time_minutes{};
};

struct RecipeRequest
{
    std::string              cuisine{"any"};
    std::string              diet{"none"};
    int                      servings{4};
    std::vector<std::string> inventory{};
    std::string              meal_type{"dinner"}; // breakfast, lunch, dinner, snack
    int                      max_prep_time{0};    // in minutes (0 = no limit)
};

} // namespace Ithaca
