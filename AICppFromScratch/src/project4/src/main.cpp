/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-26
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "llm_client/ChatClient.h"
#include "masterchef/MasterChef.h"
#include "masterchef/RecipeRequestBuilder.h"

#include <Windows.h>

#include <format>
#include <iostream>

void pretty_print_recipe(const Ithaca::Recipe &recipe)
{
    std::cout << "\n====================================\n";
    std::cout << "    " << recipe.title;
    std::cout << "\n====================================\n";

    std::cout << std::format("Prep time: {} minutes\n", recipe.prep_time_minutes);
    std::cout << std::format("Cook time: {} minutes\n", recipe.cook_time_minutes);
    std::cout << std::format("Total time: {} minutes\n", recipe.prep_time_minutes + recipe.cook_time_minutes);

    std::cout << "Ingredients\n";
    std::cout << "---------------------------\n";
    for (size_t i = 0; i < recipe.ingredients.size(); ++i)
    {
        std::cout << std::format(" {}. {}\n", i + 1, recipe.ingredients[i]);
    }

    std::cout << "\nStep\n";
    std::cout << "---------------------------\n";
    for (size_t i = 0; i < recipe.step.size(); ++i)
    {
        std::cout << std::format(" Step {}: {}\n", i + 1, recipe.step[i]);
    }

    std::cout << "\nNutrition\n";
    std::cout << "---------------------------\n";
    std::cout << std::format(" Calories {} kcal\n", recipe.nutrition.calories);
    std::cout << std::format(" protein_grams {} g\n", recipe.nutrition.protein_grams);
    std::cout << std::format(" carbs_grams {} g\n", recipe.nutrition.carbs_grams);
    std::cout << std::format(" fat_grams {} g\n", recipe.nutrition.fat_grams);

    std::cout << "\n====================================\n\n";
}

int main(int argc, const char *argv[])
{
    SetConsoleOutputCP(CP_UTF8);

    std::cout << "\n------------------------------------>";
    std::cout << "\n<-------- AI Masterchef -------->";
    std::cout << "\n------------------------------------>\n";

    try
    {
        Ithaca::ChatConfig config{
            .base  = {.provider = Ithaca::ApiProvider::OLLAMA, .api_key = "", .host = "http://127.0.0.1:11434"},
            .model = "llama3.2:3b",
            .max_output_tokens = 500
        };

        if ((Ithaca::ApiProvider::DEEPSEEK == config.base.provider
             || Ithaca::ApiProvider::OPENAI == config.base.provider)
            && config.base.api_key.empty())
        {
            throw std::runtime_error("API_KEY get unsuceesffully");
        }

        Ithaca::ChatClient chat(config);
        Ithaca::MasterChef chef(chat);

        std::cout << "\nGenerating Italian pasta recipe...\n";
        auto request
            = Ithaca::RecipeRequestBuilder().set_cuisine("Italian").set_servings(2).set_meal_type("dinner").build();
        auto recipe = chef.generate_recipe(request);
        pretty_print_recipe(recipe);

        std::cout << "\n\nGenerating Chinese recipe...\n";
        request = Ithaca::RecipeRequestBuilder().set_cuisine("Chinese").set_servings(3).set_meal_type("dinner").build();
        recipe  = chef.generate_recipe(request);
        pretty_print_recipe(recipe);
    }
    catch (const std::exception &ex)
    {
        std::cerr << "C++ std exception: " << ex.what() << '\n';
    }

    return 0;
}
