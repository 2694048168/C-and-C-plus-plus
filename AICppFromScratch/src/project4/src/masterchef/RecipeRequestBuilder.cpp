#include "masterchef/RecipeRequestBuilder.h"

#include <exception>
#include <stdexcept>

namespace Ithaca {

RecipeRequestBuilder::RecipeRequestBuilder()
    : mRequest{.cuisine       = "any",
               .diet          = "none",
               .servings      = 4,
               .inventory     = {},
               .meal_type     = "dinner",
               .max_prep_time = 0}
{
}

RecipeRequestBuilder &RecipeRequestBuilder::set_cuisine(const std::string &cuisine)
{
    mRequest.cuisine = cuisine;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_diet(const std::string &diet)
{
    mRequest.diet = diet;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_servings(int servings)
{
    if (servings <= 0)
        throw std::invalid_argument("Servings must be positive\n");

    mRequest.servings = servings;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_meal_type(const std::string &meal_type)
{
    mRequest.meal_type = meal_type;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_max_prep_time(int minutes)
{
    if (minutes < 0)
        throw std::invalid_argument("Max prep time must be positive\n");

    mRequest.max_prep_time = minutes;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_inventory(const std::vector<std::string> &items)
{
    mRequest.inventory = items;
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::set_inventory_item(const std::string &item)
{
    mRequest.inventory.emplace_back(item);
    return *this;
}

RecipeRequestBuilder &RecipeRequestBuilder::reset()
{
    mRequest
        = {.cuisine = "any", .diet = "none", .servings = 4, .inventory = {}, .meal_type = "dinner", .max_prep_time = 0};

    return *this;
}

RecipeRequest RecipeRequestBuilder::build() const
{
    return mRequest;
}

} // namespace Ithaca