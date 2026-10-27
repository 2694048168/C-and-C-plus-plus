#include "masterchef/MasterChef.h"

#include "nlohmann/json.hpp"

#include <exception>
#include <format>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace Ithaca {

MasterChef::MasterChef(ChatClient &client)
    : mClient(client)
{
}

Recipe MasterChef::generate_recipe(const RecipeRequest &request)
{
    std::string prompt = build_prompt(request);

    std::string instructions = R"(
        Return ONLY valid JSON with no markdown formatting, no code blocks,
        no explanations, and no preamble. Use this exact scheme:

        {
            "title": "Recipe Name",
            "ingredients": ["ingredient 1", "ingredient 2"],
            "steps": ["step 1", "step 2"],
            "nutrition": {
                "calories": 0,
                "protein_grams": 0,
                "carbs_grams": 0,
                "fat_grams": 0
            },
            "prep_time_minutes": 0,
            "cook_time_minutes": 0
        }
    )";

    std::string json_response = mClient.ask(prompt, instructions, 2000);

    return parse_recipe_json(json_response);
}

std::string MasterChef::build_prompt(const RecipeRequest &request)
{
    std::ostringstream prompt{};

    prompt << "Create a detailed recipe with the following requirements:\n\n";

    prompt << std::format("- Cuisine: {}", request.cuisine);

    if ("none" != request.diet)
        prompt << std::format("- Dietary restriction: {}", request.diet);

    prompt << std::format("- Servings: {}", request.servings);

    prompt << std::format("- Meal type: {}", request.meal_type);

    if (request.max_prep_time > 0)
        prompt << std::format("- Maximum preparation time: {} minutes\n", request.max_prep_time);

    if (!request.inventory.empty())
    {
        prompt << "- Must use these ingredients: ";
        for (size_t idx = 0; idx < request.inventory.size(); ++idx)
        {
            prompt << request.inventory[idx];
            if (idx < request.inventory.size() - 1)
                prompt << ", ";
        }
        prompt << "\n";
    }

    prompt << "\nProvide a complete recipe with title, ingredients list, "
           << "step-by-step instructions, nutrition information, and cooking times.";

    return prompt.str();
}

Recipe MasterChef::parse_recipe_json(const std::string &json_response)
{
    Recipe recipe{};

    try
    {
        json recipe_json = json::parse(json_response);

        recipe.title = recipe_json.at("title").get<std::string>();

        const auto &ingredients_array = recipe_json.at("ingredients");
        for (const auto &elem : ingredients_array)
        {
            recipe.ingredients.emplace_back(elem.get<std::string>());
        }

        const auto &step_array = recipe_json.at("steps");
        for (const auto &elem : step_array)
        {
            recipe.step.emplace_back(elem.get<std::string>());
        }

        const auto &nutrition          = recipe_json.at("nutrition");
        recipe.nutrition.calories      = nutrition.at("calories").get<int>();
        recipe.nutrition.protein_grams = nutrition.at("protein_grams").get<int>();
        recipe.nutrition.carbs_grams   = nutrition.at("carbs_grams").get<int>();
        recipe.nutrition.fat_grams     = nutrition.at("fat_grams").get<int>();

        recipe.prep_time_minutes = recipe_json.at("prep_time_minutes").get<int>();
        recipe.cook_time_minutes = recipe_json.at("cook_time_minutes").get<int>();

        return recipe;
    }
    catch (const json::exception &ex)
    {
        std::cerr << "JSON exception: " << ex.what() << '\n';
        std::cerr << "Failded JSON parse for recipe: " << json_response << '\n';
        return recipe;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
        return recipe;
    }
}

} // namespace Ithaca