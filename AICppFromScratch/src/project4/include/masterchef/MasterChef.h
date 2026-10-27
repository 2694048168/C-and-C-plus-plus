/**
 * @file MasterChef.h
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-27
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#pragma once

#include "llm_client/ChatClient.h"
#include "masterchef/Recipe.h"

namespace Ithaca {

class MasterChef
{
private:
    ChatClient &mClient;

private:
    std::string build_prompt(const RecipeRequest &request);
    Recipe      parse_recipe_json(const std::string &json_response);

public:
    Recipe generate_recipe(const RecipeRequest &request);

public:
    // compiler specify member-function
    explicit MasterChef(ChatClient &client);
    ~MasterChef() = default;

    MasterChef(const MasterChef &other)             = delete;
    MasterChef &operator=(const MasterChef &other)  = delete;
    MasterChef(MasterChef &&other)                  = delete;
    MasterChef       &operator=(MasterChef &&other) = delete;
    MasterChef       *operator&() {};
    const MasterChef *operator&() const {};
};

} // namespace Ithaca