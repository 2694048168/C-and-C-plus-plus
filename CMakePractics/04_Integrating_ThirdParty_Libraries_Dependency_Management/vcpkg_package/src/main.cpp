/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2024-06-14
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <rapidjson/document.h>

#include <iostream>

// ------------------------------------
int main(int argc, const char **argv)
{
    const char json[]
        = " { \"hello\" : \"world\", \"t\" : true , \"f\" : false, \"n\": null, \"i\":123, \"pi\": 3.1416, \"a\":[1, "
          "2, 3, 4] } ";

    rapidjson::Document document;
    document.Parse(json);

    if (document.HasMember("hello"))
    {
        std::cout << "Hello = " << document["hello"].GetString() << "\n";
    }
}