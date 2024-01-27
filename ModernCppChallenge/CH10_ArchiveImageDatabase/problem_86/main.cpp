/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Inserting movies into an SQLite database transactionally
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
#include <sstream>
#include <string>
#include <vector>

/**
 * @brief Inserting movies into an SQLite database transactionally
 * 
 * Extended the program written for the previous problem so that it can add new movies
 *  to the database. The movies could be read from the console, or alternatively
 *  from a text file. The insertion of movie data into several tables in the database
 *  must be performed transactionally.
 * 
 */

/**
 * @brief Solution: pngwriter library in C++
 https://github.com/pngwriter/pngwriter
------------------------------------------------------ */
std::vector<std::string> split(std::string text, const char delimiter)
{
    auto sstr   = std::stringstream{text};
    auto tokens = std::vector<std::string>{};
    auto token  = std::string{};
    while (std::getline(sstr, token, delimiter))
    {
        if (!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

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

movie read_movie()
{
    movie m;

    std::cout << "Enter movie" << std::endl;
    std::cout << "Title: ";
    std::getline(std::cin, m.title);
    std::cout << "Year: ";
    std::cin >> m.year;
    std::cout << "Length: ";
    std::cin >> m.length;
    std::cin.ignore();
    std::string directors;
    std::cout << "Directors: ";
    std::getline(std::cin, directors);
    m.directors = split(directors, ',');
    std::string writers;
    std::cout << "Writers: ";
    std::getline(std::cin, writers);
    m.writers = split(writers, ',');
    std::string cast;
    std::cout << "Cast: ";
    std::getline(std::cin, cast);
    auto roles = split(cast, ',');
    for (const auto &r : roles)
    {
        auto         pos = r.find_first_of('=');
        casting_role cr;
        cr.actor = r.substr(0, pos);
        cr.role  = r.substr(pos + 1, r.size() - pos - 1);
        m.cast.push_back(cr);
    }

    return m;
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

sqlite_int64 get_person_id(const std::string &name, sqlite::database &db)
{
    sqlite_int64 id = 0;

    db << "select rowid from persons where name=?;" << name >> [&id](const sqlite_int64 rowid)
    {
        id = rowid;
    };

    return id;
}

sqlite_int64 insert_person(std::string_view name, sqlite::database &db)
{
    db << "insert into persons values(?);" << name.data();
    return db.last_insert_rowid();
}

void insert_directors(const sqlite_int64 movie_id, const std::vector<std::string> &directors, sqlite::database &db)
{
    for (const auto &director : directors)
    {
        auto id = get_person_id(director, db);

        if (id == 0)
            id = insert_person(director, db);

        db << "insert into directors values(?, ?);" << movie_id << id;
    }
}

void insert_writers(const sqlite_int64 movie_id, const std::vector<std::string> &writers, sqlite::database &db)
{
    for (const auto &writer : writers)
    {
        auto id = get_person_id(writer, db);

        if (id == 0)
            id = insert_person(writer, db);

        db << "insert into writers values(?, ?);" << movie_id << id;
    }
}

void insert_cast(const sqlite_int64 movie_id, const std::vector<casting_role> &cast, sqlite::database &db)
{
    for (const auto &cr : cast)
    {
        auto id = get_person_id(cr.actor, db);

        if (id == 0)
            id = insert_person(cr.actor, db);

        db << "insert into casting values(?,?,?);" << movie_id << id << cr.role;
    }
}

void insert_movie(movie &m, sqlite::database &db)
{
    try
    {
        db << "begin;";

        db << "insert into movies values(?,?,?);" << m.title << m.year << m.length;

        auto movieid = db.last_insert_rowid();

        insert_directors(movieid, m.directors, db);
        insert_writers(movieid, m.writers, db);
        insert_cast(movieid, m.cast, db);

        m.id = static_cast<unsigned int>(movieid);

        db << "commit;";
    }
    catch (const std::exception &)
    {
        db << "rollback;";
    }
}

// ------------------------------
int main(int argc, char **argv)
{
    try
    {
        sqlite::database db(R"(cppchallenger86.db)");

        auto movie = read_movie();
        insert_movie(movie, db);

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
