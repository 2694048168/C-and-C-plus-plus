/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Reading movies from an SQLite database
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "movies.h"
#include "sqlite3.h"
#include "sqlite_modern_cpp.h"

#include <iostream>
#include <vector>

/**
 * @brief Reading movies from an SQLite database
 * 
 * Write a program that reads movies from an SQLite database and displays them on the console.
 * Each movie must have a numerical identifier, a title, release year, length in minutes, 
 * list of directors, list of writers, and a cast that includes both the actor and the character names.
 * 
 */

/**
 * @brief Solution: SQL-lite library
 https://www.sqlite.org/index.html
 https://github.com/SqliteModernCpp/sqlite_modern_cpp
------------------------------------------------------ */
void print_movie(const movie &m)
{
    std::cout << "[" << m.id << "] " << m.title << " (" << m.year << ") " << m.length << "min" << std::endl;
    std::cout << " directed by: ";
    for (const auto &d : m.directors) std::cout << d << ",";
    std::cout << std::endl;
    std::cout << " written by: ";
    for (const auto &w : m.writers) std::cout << w << ",";
    std::cout << std::endl;
    std::cout << " cast: ";
    for (const auto &r : m.cast) std::cout << r.actor << " (" << r.role << "),";
    std::cout << std::endl << std::endl;
}

std::vector<std::string> get_directors(const sqlite3_int64 movie_id, sqlite::database &db)
{
    std::vector<std::string> result;
    db << R"(select p.name from directors as d 
            join persons as p on d.personid = p.rowid 
            where d.movieid = ?;)"
       << movie_id
        >> [&result](const std::string name)
    {
        result.emplace_back(name);
    };

    return result;
}

std::vector<std::string> get_writers(const sqlite3_int64 movie_id, sqlite::database &db)
{
    std::vector<std::string> result;
    db << R"(select p.name from writers as w
         join persons as p on w.personid = p.rowid 
         where w.movieid = ?;)"
       << movie_id
        >> [&result](const std::string name)
    {
        result.emplace_back(name);
    };

    return result;
}

std::vector<casting_role> get_cast(const sqlite3_int64 movie_id, sqlite::database &db)
{
    std::vector<casting_role> result;
    db << R"(select p.name, c.role from casting as c
         join persons as p on c.personid = p.rowid
         where c.movieid = ?;)"
       << movie_id
        >> [&result](const std::string name, std::string role)
    {
        result.emplace_back(casting_role{name, role});
    };

    return result;
}

movie_list get_movies(sqlite::database &db)
{
    movie_list movies;

    db << R"(select rowid, * from movies;)" >>
        [&movies, &db](const sqlite3_int64 rowid, const std::string &title, const int year, const int length)
    {
        movies.emplace_back(movie{static_cast<unsigned int>(rowid), title, year, static_cast<unsigned int>(length),
                                  get_cast(rowid, db), get_directors(rowid, db), get_directors(rowid, db)});
    };

    return movies;
}

// ------------------------------
int main(int argc, char **argv)
{
    try
    {
        sqlite::database db(R"(cppchallenger85.db)");

        auto movies = get_movies(db);
        for (const auto &m : movies) print_movie(m);
    }
    catch (const sqlite::sqlite_exception &e)
    {
        std::cerr << e.get_code() << ": " << e.what() << " during " << e.get_sql() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
