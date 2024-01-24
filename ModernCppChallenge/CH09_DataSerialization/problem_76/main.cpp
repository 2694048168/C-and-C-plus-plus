/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Deserializing data from JSON
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "movies.h"
#include "nlohmannjson/json.hpp"

#include <assert.h>

#include <fstream>
#include <iostream>
#include <string_view>

/**
 * @brief Deserializing data from JSON
 * 
 * Consider a JSON file with a list of movies as shown in the previous problem.
 * Write a program that can deserialize its content.
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
using json = nlohmann::json;

movie_list deserialize(std::string_view filepath)
{
    movie_list movies;

    std::ifstream ifile(filepath.data());
    if (ifile.is_open())
    {
        json jdata;

        try
        {
            ifile >> jdata;

            if (jdata.is_object())
            {
                for (auto &element : jdata.at("movies"))
                {
                    movie m;

                    m.id     = element.at("id").get<unsigned int>();
                    m.title  = element.at("title").get<std::string>();
                    m.year   = element.at("year").get<unsigned int>();
                    m.length = element.at("length").get<unsigned int>();

                    for (auto &role : element.at("cast"))
                    {
                        m.cast.push_back(
                            casting_role{role.at("star").get<std::string>(), role.at("name").get<std::string>()});
                    }

                    for (auto &director : element.at("directors"))
                    {
                        m.directors.push_back(director);
                    }

                    for (auto &writer : element.at("writers"))
                    {
                        m.writers.push_back(writer);
                    }

                    movies.push_back(std::move(m));
                }
            }
        }
        catch (const std::exception &ex)
        {
            std::cout << ex.what() << std::endl;
        }
    }

    return movies;
}

// ------------------------------
int main(int argc, char **argv)
{
    auto movies = deserialize("movies.json");

    assert(movies.size() == 2);
    assert(movies[0].title == "The Matrix");
    assert(movies[1].title == "Forrest Gump");

    return 0;
}
