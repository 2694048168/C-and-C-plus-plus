/**
 * @file main.cpp
 * @author Wei Li (Ithaca) (weili_yzzca@163.com)
 * @brief 
 * @version 0.1
 * @date 2026-10-23
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#include "AI_json.hpp"
#include "accessing_json.hpp"
#include "basic.hpp"
#include "iterating_json.hpp"
#include "magic_macro.hpp"
#include "nested_objects.hpp"
#include "safely_accessing.hpp"

#include <iostream>

int main(int argc, const char *argv[])
{
    std::cout << "---->[Begin] The Tutorial for JSON\n\n";

    Ithaca::BasicExample();
    Ithaca::NestedExample();
    Ithaca::AIExample();
    Ithaca::AccessingExample();
    Ithaca::SafelyExample();
    Ithaca::IteratingExample();
    Ithaca::JsonMacroExample();

    std::cout << "---->[End] The Tutorial for JSON\n\n";

    return 0;
}
