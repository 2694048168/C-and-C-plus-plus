/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief Handling movie images in an SQLite database
 * @version 0.1
 * @date 2024-01-27
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "sqlite3.h"

#define MODERN_SQLITE_STD_OPTIONAL_SUPPORT
#include "movies.h"
#include "sqlite_modern_cpp.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using std::optional;

/**
 * @brief Handling movie images in an SQLite database
 * 
 * Modify the program written for the previous problem to support adding 
 * media files (such as images, but also videos) to a movie. These files must be stored
 *  in a separate table in the database and have a unique numerical identifier, 
 * the movie identifier, a name (typically the filename), an optional description,
 * and the actual media content, stored as a blob. 
 * 
 */

/**
 * @brief Solution: pngwriter library in C++
 https://github.com/pngwriter/pngwriter
------------------------------------------------------ */
movie_list get_movies(std::string_view title, sqlite::database &db)
{
    movie_list movies;

    db << R"(select rowid, * from movies where title=?;)" << title.data() >>
        [&movies, &db](const sqlite3_int64 rowid, const std::string &title, const int year, const int length)
    {
        movies.emplace_back(
            movie{static_cast<unsigned int>(rowid), title, year, static_cast<unsigned int>(length), {}, {}, {}});
    };

    return movies;
}

bool add_media(const sqlite_int64 movieid, std::string_view name, std::string_view description,
               std::vector<char> content, sqlite::database &db)
{
    try
    {
        db << "insert into media values(?,?,?,?)" << movieid << name.data() << description.data() << content;

        return true;
    }
    catch (...)
    {
        return false;
    }
}

media_list get_media(const sqlite_int64 movieid, sqlite::database &db)
{
    media_list list;

    db << "select rowid, * from media where movieid = ?;" << movieid >>
        [&list](const sqlite_int64 rowid, const sqlite_int64 movieid, const std::string &name,
#ifdef USE_BOOST_OPTIONAL
                const std::unique_ptr<std::string> text,
#else
                const optional<std::string> text,
#endif
                const std::vector<char> &blob)
    {
        list.emplace_back(media{static_cast<unsigned int>(rowid), static_cast<unsigned int>(movieid), name,
#ifdef USE_BOOST_OPTIONAL
                                text != nullptr ? *text : optional<std::string>{},
#else
                                text,
#endif
                                blob});
    };

    return list;
}

bool delete_media(const sqlite_int64 mediaid, sqlite::database &db)
{
    try
    {
        db << "delete from media where rowid = ?;" << mediaid;

        return true;
    }
    catch (...)
    {
        return false;
    }
}

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

inline bool starts_with(std::string_view text, std::string_view part)
{
    return text.find(part) == 0;
}

inline std::string trim(std::string_view text)
{
    auto first{text.find_first_not_of(' ')};
    auto last{text.find_last_not_of(' ')};
    return text.substr(first, (last - first + 1)).data();
}

std::vector<char> load_image(std::string_view filepath)
{
    std::vector<char> data;

    std::ifstream ifile(filepath.data(), std::ios::binary | std::ios::ate);
    if (ifile.is_open())
    {
        auto size = ifile.tellg();
        ifile.seekg(0, std::ios::beg);

        data.resize(static_cast<size_t>(size));
        ifile.read(reinterpret_cast<char *>(data.data()), size);
    }

    return data;
}

void run_find(std::string_view line, sqlite::database &db)
{
    auto title = trim(line.substr(5));

    auto movies = get_movies(title, db);
    if (movies.empty())
        std::cout << "empty" << std::endl;
    else
    {
        for (const auto m : movies)
        {
            std::cout << m.id << " | " << m.title << " | " << m.year << " | " << m.length << "min" << std::endl;
        }
    }
}

void run_list(std::string_view line, sqlite::database &db)
{
    auto movieid = std::stoi(trim(line.substr(5)));
    if (movieid > 0)
    {
        auto list = get_media(movieid, db);
        if (list.empty())
        {
            std::cout << "empty" << std::endl;
        }
        else
        {
            for (const auto &m : list)
            {
                std::cout << m.id << " | " << m.movie_id << " | " << m.name << " | " << m.text.value_or("(null)")
                          << " | " << m.blob.size() << " bytes" << std::endl;
            }
        }
    }
    else
        std::cout << "input error" << std::endl;
}

void run_add(std::string_view line, sqlite::database &db)
{
    auto parts = split(trim(line.substr(4)), ',');
    if (parts.size() == 3)
    {
        auto movieid = std::stoi(parts[0]);
        auto path    = fs::path{parts[1]};
        auto desc    = parts[2];

        auto content = load_image(parts[1]);
        auto name    = path.filename().string();

        auto success = add_media(movieid, name, desc, content, db);
        if (success)
            std::cout << "added" << std::endl;
        else
            std::cout << "failed" << std::endl;
    }
    else
        std::cout << "input error" << std::endl;
}

void run_del(std::string_view line, sqlite::database &db)
{
    auto mediaid = std::stoi(trim(line.substr(4)));
    if (mediaid > 0)
    {
        auto success = delete_media(mediaid, db);
        if (success)
            std::cout << "deleted" << std::endl;
        else
            std::cout << "failed" << std::endl;
    }
    else
        std::cout << "input error" << std::endl;
}

void print_commands()
{
    std::cout << "find <title>                        finds a movie ID\n"
              << "list <movieid>                      lists the images of a movie\n"
              << "add <movieid>,<path>,<description>  adds a new image\n"
              << "del <imageid>                       delete an image\n"
              << "help                                shows available commands\n"
              << "exit                                exists the application\n";
}

// ------------------------------
int main(int argc, char **argv)
{
    try
    {
        sqlite::database db(R"(cppchallenger87.db)");

        while (true)
        {
            std::string line;
            std::getline(std::cin, line);

            if (line == "help")
                print_commands();
            else if (line == "exit")
                break;
            else
            {
                if (starts_with(line, "find"))
                    run_find(line, db);
                else if (starts_with(line, "list"))
                    run_list(line, db);
                else if (starts_with(line, "add"))
                    run_add(line, db);
                else if (starts_with(line, "del"))
                    run_del(line, db);
                else
                    std::cout << "unknown command" << std::endl;
            }

            std::cout << std::endl;
        }
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
