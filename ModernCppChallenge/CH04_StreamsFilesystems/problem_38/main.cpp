/**
 * @file main.cpp
 * @author Wei Li (weili_yzzcq@163.com)
 * @brief 
 * @version 0.1
 * @date 2023-11-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

/**
 * @brief Temporary log files
 * 
 * Create a logging class that writes text messages to a discardable text file. 
 * The text file should have a unique name and must be located in a temporary directory. 
 * Unless specified otherwise, this log file should be deleted 
 * when the instance of the class is destroyed.
 * However, it should be possible to retain the log file 
 * by moving it to a permanent location.
 * ---------------------------------------------------------*/

/**
 * @brief Solution:

The logging class that you have to implement for this task should:
1. Have a constructor that creates a text file in a temporary directory
 and opens it for writing
2. During destruction, if the file still exists, close and delete it
3. Have a method that closes the file and moves it to a permanent path
4. Overloads operator<< to write a text message to the output file
------------------------------------------------------ */
class logger
{
    std::filesystem::path log_path;
    std::ofstream         log_file;

public:
    logger()
    {
        // auto name = uuids::to_string(uuids::uuid_random_generator{}());
        std::string name = "test";
        log_path         = std::filesystem::temp_directory_path() / (name + ".tmp");
        log_file.open(log_path.c_str(), std::ios::out | std::ios::trunc);
    }

    ~logger() noexcept
    {
        try
        {
            if (log_file.is_open())
                log_file.close();
            if (!log_path.empty())
                std::filesystem::remove(log_path);
        }
        catch (...)
        {
        }
    }

    void persist(const std::filesystem::path &path)
    {
        log_file.close();
        std::filesystem::rename(log_path, path);
        log_path.clear();
    }

    logger &operator<<(std::string_view message)
    {
        log_file << message.data() << '\n';
        return *this;
    }
};

// ------------------------------
int main(int argc, char **argv)
{
    logger log;
    try
    {
        log << "this is a line"
            << "and this is another one";

        throw std::runtime_error("error");
    }
    catch (...)
    {
        log.persist(R"(lastlog.txt)");
    }

    return 0;
}