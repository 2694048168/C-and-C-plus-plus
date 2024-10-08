/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Serializing data to JSON
 * @version 0.1
 * @date 2024-01-23
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "movies.h"
#include "nlohmannjson/json.hpp"

#include <fstream>
#include <iostream>
#include <string_view>

using json = nlohmann::json;

/**
 * @brief Serializing data to JSON
 * 
 * Write a program that can serialize a list of movies, 
 * as defined for the previous problems, to a JSON file. 
 * Each movie has a numerical identifier, title, release year, length in minutes,
 * a list of directors, a list of writers, and a list of casting roles with actor name
 *  and character name. 
 * 
 */

/**
 * @brief Solution:
------------------------------------------------------ */
void to_json(json &j, const casting_role &c)
{
    j = json{
        {"star", c.actor},
        {"name",  c.role}
    };
}

void to_json(json &j, const movie &m)
{
    j = json::object({
        {       "id",        m.id},
        {    "title",     m.title},
        {     "year",      m.year},
        {   "length",    m.length},
        {     "cast",      m.cast},
        {"directors", m.directors},
        {  "writers",   m.writers}
    });
}

void serialize(const movie_list &movies, std::string_view filepath)
{
    json jdata{
        {"movies", movies}
    };

    std::ofstream ofile(filepath.data());
    if (ofile.is_open())
    {
        ofile << std::setw(2) << jdata << std::endl;
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    movie_list movies{
        {
         11001,   "The Matrix",
         1999, 196,
         {{"Keanu Reeves", "Neo"},
         {"Laurence Fishburne", "Morpheus"},
         {"Carrie-Anne Moss", "Trinity"},
         {"Hugo Weaving", "Agent Smith"}},
         {"Lana Wachowski", "Lilly Wachowski"},
         {"Lana Wachowski", "Lilly Wachowski"},
         },
        {
         9871, "Forrest Gump",
         1994, 202,
         {{"Tom Hanks", "Forrest Gump"},
         {"Sally Field", "Mrs. Gump"},
         {"Robin Wright", "Jenny Curran"},
         {"Mykelti Williamson", "Bubba Blue"}},
         {"Robert Zemeckis"},
         {"Winston Groom", "Eric Roth"},
         }
    };

    serialize(movies, "movies.json");

    return 0;
}
