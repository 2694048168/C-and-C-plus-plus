/**
 * @file RecipeRequestBuilder.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "masterchef/Recipe.h"

namespace Ithaca {

class RecipeRequestBuilder
{
private:
    RecipeRequest mRequest;

public:
    // build methods - return reference for chaining
    RecipeRequestBuilder &set_cuisine(const std::string &cuisine);
    RecipeRequestBuilder &set_diet(const std::string &diet);
    RecipeRequestBuilder &set_servings(int servings);
    RecipeRequestBuilder &set_meal_type(const std::string &meal_type);

    RecipeRequestBuilder &set_max_prep_time(int minutes);
    RecipeRequestBuilder &set_inventory(const std::vector<std::string> &items);
    RecipeRequestBuilder &set_inventory_item(const std::string& item);
    RecipeRequestBuilder &reset();

    RecipeRequest build() const;

public:
    // compiler specify member-function
    explicit RecipeRequestBuilder();
    ~RecipeRequestBuilder()         = default;

    RecipeRequestBuilder(const RecipeRequestBuilder &other)             = delete;
    RecipeRequestBuilder &operator=(const RecipeRequestBuilder &other)  = delete;
    RecipeRequestBuilder(RecipeRequestBuilder &&other)                  = delete;
    RecipeRequestBuilder       &operator=(RecipeRequestBuilder &&other) = delete;
    RecipeRequestBuilder       *operator&() {};
    const RecipeRequestBuilder *operator&() const {};
};

} // namespace Ithaca